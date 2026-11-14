#!/usr/bin/env python3
"""merge — 本地双端短读段合并命令。

把同一短片段的两端读段（R1/R2）合成一条共识序列。范围严格受限：
仅支持 Phred+33 质量、ACGTN 字母表、无 indel 的末端重叠；
不做参考比对、接头学习或变异检测。

用法：
    python3 merge.py [--r1 fixtures/reads_R1.fastq] [--r2 fixtures/reads_R2.fastq]
                     [--plan fixtures/plan.json] [--outdir out]

产物（outdir 下）：
    merged.fastq     仅位置可确定配对的共识记录（记录名 = 去掉 mate 后缀的配对键）
    unmerged.fastq   未合并配对与孤儿的原始两端记录（原方向、原名，不伪造完整片段）
    pairs.tsv        逐配对状态与最佳/备选候选依据
    candidates.tsv   全部候选重叠的评分与拒绝原因
    provenance.tsv   merged.fastq 每个输出位置的来源（mate、碱基索引、是否融合）
    report.json      以上全部内容的机器可读汇总 + 参数与诊断
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ALPHABET = set("ACGTN")
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
QUAL_ASCII_MIN, QUAL_ASCII_MAX = 33, 126  # Phred+33 可打印范围
HQ_MIN_Q = 20  # 高质量矛盾阈值：两条观测 Q 均 >= 20 且碱基不同，视为硬冲突
TOOL_VERSION = "1.0.0"

STATUS_MERGED = "merged"            # 唯一兼容候选且支持充分
STATUS_AMBIGUOUS = "ambiguous"      # 多个同等兼容位置，位置不可识别
STATUS_NO_OVERLAP = "no_overlap"    # 无兼容候选
STATUS_ORPHAN = "orphan"            # 缺少另一端


# ---------------------------------------------------------------- 基础工具

def err_prob(q: int) -> float:
    """plan 模型：观测到非 N 碱基的错误概率为 10**(-Q/10)。"""
    return 10.0 ** (-q / 10.0)


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def phred_char(q: int) -> str:
    return chr(q + 33)


# ---------------------------------------------------------------- 输入解析

@dataclass
class Record:
    name: str       # 完整名字（首个空白前），含 mate 后缀
    key: str        # 去掉明确 mate 后缀后的配对键
    suffix: str     # 实际去掉的后缀（"" 表示没有明确后缀）
    seq: str
    qual: str
    file: str
    line: int       # '@' 所在行（1 基）


def read_fastq(path: Path, diagnostics: list[dict]) -> list[Record]:
    """读取并逐记录校验 FASTQ；坏记录保留来源诊断后剔除。"""
    lines = path.read_text().splitlines()
    records: list[Record] = []
    if len(lines) % 4 != 0:
        diagnostics.append({
            "file": str(path), "line": len(lines) - len(lines) % 4 + 1,
            "name": "", "reason": f"文件行数 {len(lines)} 不是 4 的整数倍，末尾不完整记录已忽略",
        })
    for i in range(0, len(lines) - len(lines) % 4, 4):
        head, seq, plus, qual = lines[i:i + 4]
        name = head[1:].split()[0] if head.startswith("@") and head[1:].split() else ""
        problems = []
        if not head.startswith("@"):
            problems.append("记录首行缺少 '@'")
        elif not name:
            problems.append("记录名为空")
        if not plus.startswith("+"):
            problems.append("记录第三行缺少 '+'")
        if len(seq) != len(qual):
            problems.append(f"序列长度 {len(seq)} 与质量长度 {len(qual)} 不一致")
        bad_bases = sorted(set(seq) - ALPHABET)
        if bad_bases:
            problems.append(f"序列含非法碱基 {bad_bases}（仅支持 ACGTN）")
        if any(not QUAL_ASCII_MIN <= ord(c) <= QUAL_ASCII_MAX for c in qual):
            problems.append("质量字符超出 Phred+33 可打印范围 [33,126]")
        if problems:
            diagnostics.append({
                "file": str(path), "line": i + 1, "name": name,
                "reason": "；".join(problems),
            })
            continue
        records.append(Record(name=name, key=name, suffix="", seq=seq,
                              qual=qual, file=str(path), line=i + 1))
    return records


def strip_mate_suffix(records: list[Record], suffixes: list[str]) -> None:
    """按 plan 给出的明确 mate 后缀剥离名字，得到配对键。"""
    for rec in records:
        for suf in suffixes:
            if rec.name.endswith(suf) and len(rec.name) > len(suf):
                rec.key = rec.name[: -len(suf)]
                rec.suffix = suf
                break


def pair_records(r1_recs: list[Record], r2_recs: list[Record],
                 diagnostics: list[dict]) -> tuple[dict[str, dict[str, Record]], list[Record]]:
    """小输入允许内存配对：按键分组，校验身份唯一，孤儿保留诊断。"""
    pairs: dict[str, dict[str, Record]] = {}
    orphans: list[Record] = []

    def index_by_key(recs: list[Record]) -> dict[str, Record]:
        seen: dict[str, list[Record]] = {}
        for rec in recs:
            seen.setdefault(rec.key, []).append(rec)
        out = {}
        for key, group in seen.items():
            if len(group) > 1:
                for rec in group:
                    diagnostics.append({
                        "file": rec.file, "line": rec.line, "name": rec.name,
                        "reason": f"键 '{key}' 在同一文件中出现 {len(group)} 次，身份不唯一，全部剔除",
                    })
            else:
                out[key] = group[0]
        return out

    r1_by_key = index_by_key(r1_recs)
    r2_by_key = index_by_key(r2_recs)
    for key in sorted(set(r1_by_key) | set(r2_by_key)):
        a, b = r1_by_key.get(key), r2_by_key.get(key)
        if a and b:
            pairs[key] = {"R1": a, "R2": b}
        else:
            lone = a or b
            orphans.append(lone)
            diagnostics.append({
                "file": lone.file, "line": lone.line, "name": lone.name,
                "reason": f"键 '{key}' 缺少另一端记录，作为孤儿保留",
            })
    return pairs, orphans


# ---------------------------------------------------------------- 重叠候选

@dataclass
class Candidate:
    orientation: str   # "standard"=R1后缀×R2rc前缀；"dovetail"=R1前缀×R2rc后缀
    d: int             # R2rc 起点在 R1 坐标系中的位置（d<0 为 dovetail）
    overlap: int
    n_informative: int = 0     # 双端均非 N 的比较位点数
    n_match: int = 0
    n_mismatch: int = 0
    score: float = 0.0         # 对数似然比支持分数（自然对数）
    hq_conflicts: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    compatible: bool = False

    def to_dict(self, full: bool = False) -> dict:
        d = {
            "orientation": self.orientation, "d": self.d, "overlap": self.overlap,
            "n_informative": self.n_informative, "n_match": self.n_match,
            "n_mismatch": self.n_mismatch, "score": round(self.score, 6),
            "hq_conflict_count": len(self.hq_conflicts),
            "rejected": self.rejected, "compatible": self.compatible,
        }
        # 冲突位置明细只对最佳/兼容候选保留，其余候选给计数，避免报告膨胀
        d["hq_conflicts"] = self.hq_conflicts if full else []
        return d


def position_llr(b1: str, q1: int, b2: str, q2: int) -> float:
    """单个重叠位点的对数似然比：同一位点 vs 两条独立随机序列。

    零假设下每个非 N 观测概率 1/4、N 概率 1；N 不提供确定碱基证据，贡献 0。
    """
    if b1 == "N" or b2 == "N":
        return 0.0
    e1, e2 = err_prob(q1), err_prob(q2)
    if b1 == b2:
        p = (1 - e1) * (1 - e2) + e1 * e2 / 3.0
    else:
        p = (e2 * (1 - e1) + e1 * (1 - e2)) / 3.0 + 2.0 * e1 * e2 / 9.0
    return math.log(4.0 * p)


def enumerate_candidates(seq1: str, q1: list[int], seq2rc: str, q2rc: list[int],
                         min_ov: int, max_ov: int, hq_min_q: int) -> list[Candidate]:
    """枚举允许长度内全部无 indel 末端重叠并打分、应用拒绝规则。"""
    len1, len2 = len(seq1), len(seq2rc)
    candidates: list[Candidate] = []
    # d = R2rc 起点在 R1 坐标系中的位置；遍历所有能产生末端重叠的位移
    for d in range(-(len2 - min_ov), len1 - min_ov + 1):
        i2_lo = max(0, -d)
        i2_hi = min(len2, len1 - d)
        overlap = i2_hi - i2_lo
        if overlap < min_ov or overlap > max_ov:
            continue
        i1_lo, i1_end = i2_lo + d, i2_hi + d
        # 末端重叠：重叠区必须触及某条读段的 3' 端与另一条的 5' 端；
        # 一条完全包含另一条不属于末端重叠，跳过
        terminal = (i2_lo == 0 and i1_end == len1) or (i1_lo == 0 and i2_hi == len2)
        if not terminal:
            continue
        cand = Candidate(orientation="standard" if d >= 0 else "dovetail",
                         d=d, overlap=overlap)
        for k in range(i2_lo, i2_hi):
            i1 = k + d
            b1, b2 = seq1[i1], seq2rc[k]
            qq1, qq2 = q1[i1], q2rc[k]
            cand.score += position_llr(b1, qq1, b2, qq2)
            if b1 != "N" and b2 != "N":
                cand.n_informative += 1
                if b1 == b2:
                    cand.n_match += 1
                else:
                    cand.n_mismatch += 1
                    if qq1 >= hq_min_q and qq2 >= hq_min_q:
                        cand.hq_conflicts.append({
                            "r1_idx": i1, "r2_idx": len2 - 1 - k,
                            "r1_base": b1, "r1_q": qq1,
                            "r2_base": b2, "r2_q": qq2,
                        })
        # 拒绝规则（可解释，全部记录）：
        if cand.score <= 0.0:
            cand.rejected.append(f"支持分数 {cand.score:.3f} <= 0，不优于两条独立随机序列")
        if cand.hq_conflicts:
            shown = "、".join(f"R1:{c['r1_idx']} {c['r1_base']}(Q{c['r1_q']})"
                              f" vs R2:{c['r2_idx']} {c['r2_base']}(Q{c['r2_q']})"
                              for c in cand.hq_conflicts[:3])
            more = f" …等 {len(cand.hq_conflicts)} 处" if len(cand.hq_conflicts) > 3 else ""
            cand.rejected.append(f"含 {len(cand.hq_conflicts)} 处高质量矛盾（{shown}{more}）")
        cand.compatible = not cand.rejected
        candidates.append(cand)
    return candidates


# ---------------------------------------------------------------- 共识融合

def fuse_two(b1: str, q1: int, b2: str, q2: int, cap: int) -> tuple[str, int, str]:
    """按 plan 的独立错误概率假设融合同一位置的两条观测。

    返回 (共识碱基, Phred 质量, 说明)。后验 P(t) ∝ P(b1|t)P(b2|t)（均匀先验），
    质量 = -10*log10(1 - 最大后验)，四舍五入（half-up）并以 cap 为上界。
    """
    obs = [(b, q) for b, q in ((b1, q1), (b2, q2)) if b != "N"]
    if not obs:
        return "N", 0, "both_N"            # 无任何确定碱基证据：N，质量 0
    if len(obs) == 1:
        b, q = obs[0]
        return b, min(q, cap), "single_non_N"  # N 不可知，唯一观测原样保留
    (ba, qa), (bb, qb) = obs
    ea, eb = err_prob(qa), err_prob(qb)
    lik = {}
    for t in "ACGT":
        pa = (1.0 - ea) if ba == t else ea / 3.0
        pb = (1.0 - eb) if bb == t else eb / 3.0
        lik[t] = pa * pb
    top = max(lik.values())
    total = sum(lik.values())
    q = min(cap, int(-10.0 * math.log10(1.0 - top / total) + 0.5))
    winners = [t for t, v in lik.items() if v == top]
    if len(winners) > 1:
        return "N", q, "tie_to_N"          # 后验并列，证据不足：N
    return winners[0], q, ("agree" if ba == bb else "conflict_resolved")


def build_consensus(rec1: Record, rec2: Record, cand: Candidate,
                    cap: int) -> tuple[str, str, list[dict]]:
    """对位置唯一确定的配对生成共识与逐位置来源表。"""
    seq1, seq2rc = rec1.seq, revcomp(rec2.seq)
    q1 = [ord(c) - 33 for c in rec1.qual]
    q2rc = [ord(c) - 33 for c in rec2.qual[::-1]]
    len1, len2 = len(seq1), len(seq2rc)
    d = cand.d
    off1, off2 = (max(0, -d), max(0, d))  # 两条读段在输出坐标系中的起点
    out_len = max(off1 + len1, off2 + len2)
    bases, quals, provenance = [], [], []
    for p in range(out_len):
        in1 = off1 <= p < off1 + len1
        in2 = off2 <= p < off2 + len2
        i1 = p - off1 if in1 else None
        i2rc = p - off2 if in2 else None
        i2 = len2 - 1 - i2rc if in2 else None  # 原始 R2 方向中的碱基索引
        prov = {"pos": p, "r1_idx": i1, "r2_idx": i2,
                "r1_base": seq1[i1] if in1 else "", "r1_q": q1[i1] if in1 else None,
                "r2_base": seq2rc[i2rc] if in2 else "", "r2_q": q2rc[i2rc] if in2 else None}
        if in1 and in2:
            base, q, note = fuse_two(seq1[i1], q1[i1], seq2rc[i2rc], q2rc[i2rc], cap)
            prov.update(source="BOTH", fused=True, note=note)
        elif in1:
            base, q = seq1[i1], q1[i1]
            prov.update(source="R1", fused=False, note="r1_only")
        else:
            base, q = seq2rc[i2rc], q2rc[i2rc]
            prov.update(source="R2", fused=False, note="r2_only")
        prov.update(consensus_base=base, consensus_q=q)
        bases.append(base)
        quals.append(phred_char(q))
        provenance.append(prov)
    return "".join(bases), "".join(quals), provenance


# ---------------------------------------------------------------- 配对决策

def decide_pair(key: str, rec1: Record, rec2: Record, plan: dict,
                hq_min_q: int) -> dict:
    """评估一对读段：R2 反向互补+质量反转，枚举候选，判定状态。"""
    seq2rc = revcomp(rec2.seq)
    q1 = [ord(c) - 33 for c in rec1.qual]
    q2rc = [ord(c) - 33 for c in rec2.qual[::-1]]
    candidates = enumerate_candidates(rec1.seq, q1, seq2rc, q2rc,
                                      plan["min_overlap"], plan["max_overlap"],
                                      hq_min_q)
    compatible = [c for c in candidates if c.compatible]
    best = None
    if candidates:
        # 最佳候选仅用于报告参照；绝不用它打破兼容并列（重复序列不可识别性）
        best = max(candidates, key=lambda c: (c.score, c.overlap, -abs(c.d)))

    result = {
        "key": key, "status": "", "reason": "",
        "r1": {"name": rec1.name, "length": len(rec1.seq)},
        "r2": {"name": rec2.name, "length": len(rec2.seq)},
        "candidates": candidates, "compatible": compatible, "best": best,
        "consensus_seq": None, "consensus_qual": None, "provenance": None,
    }
    if len(compatible) == 1:
        c = compatible[0]
        seq, qual, prov = build_consensus(rec1, rec2, c, plan["consensus_quality_cap"])
        result.update(status=STATUS_MERGED, consensus_seq=seq,
                      consensus_qual=qual, provenance=prov,
                      reason=(f"唯一兼容候选：{c.orientation} d={c.d} "
                              f"overlap={c.overlap} score={c.score:.3f}"))
    elif len(compatible) > 1:
        offsets = "、".join(f"{c.orientation}:d={c.d},L={c.overlap}" for c in compatible)
        result.update(status=STATUS_AMBIGUOUS,
                      reason=(f"{len(compatible)} 个同等兼容的位移（{offsets}）；"
                              f"重复/低复杂度序列使位置不可识别，"
                              f"较长重叠仅因累积更多匹配而分数更高，不能据此定为唯一，"
                              f"保留原始两端不生成共识"))
    else:
        if best is None:
            reason = (f"读段长度不足以达到 min_overlap={plan['min_overlap']}，"
                      f"无可评估候选")
        else:
            reason = (f"无兼容候选；最接近者 {best.orientation} d={best.d} "
                      f"overlap={best.overlap} score={best.score:.3f} 被拒绝："
                      f"{'；'.join(best.rejected)}")
        result.update(status=STATUS_NO_OVERLAP, reason=reason)
    return result


# ---------------------------------------------------------------- 管线

def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    required = ["encoding", "mate_name_suffixes", "min_overlap",
                "max_overlap", "consensus_quality_cap"]
    missing = [k for k in required if k not in plan]
    if missing:
        raise SystemExit(f"plan 缺少必需键: {missing}")
    if plan["encoding"] != "Phred+33":
        raise SystemExit(f"仅支持 Phred+33，plan 给出: {plan['encoding']}")
    if plan.get("no_indels") is not True:
        raise SystemExit("仅支持无 indel 的末端重叠（plan.no_indels 应为 true）")
    if not (0 < plan["min_overlap"] <= plan["max_overlap"]):
        raise SystemExit("plan 中 min_overlap/max_overlap 不合法")
    return plan


def run_pipeline(r1_path: Path, r2_path: Path, plan: dict,
                 hq_min_q: int = HQ_MIN_Q) -> dict:
    diagnostics: list[dict] = []
    r1_recs = read_fastq(r1_path, diagnostics)
    r2_recs = read_fastq(r2_path, diagnostics)
    strip_mate_suffix(r1_recs, plan["mate_name_suffixes"])
    strip_mate_suffix(r2_recs, plan["mate_name_suffixes"])
    pairs, orphans = pair_records(r1_recs, r2_recs, diagnostics)

    pair_results = {key: decide_pair(key, pr["R1"], pr["R2"], plan, hq_min_q)
                    for key, pr in pairs.items()}
    for key, pr in pair_results.items():  # 原始记录引用，供 unmerged.fastq 原样输出
        pr["_records"] = pairs[key]
    params = {
        "tool_version": TOOL_VERSION,
        "encoding": plan["encoding"],
        "mate_name_suffixes": plan["mate_name_suffixes"],
        "min_overlap": plan["min_overlap"],
        "max_overlap": plan["max_overlap"],
        "consensus_quality_cap": plan["consensus_quality_cap"],
        "hq_min_q": hq_min_q,
        "no_indels": plan.get("no_indels"),
        "error_model": plan.get("model"),
        "score": "sum of per-position log-likelihood ratios (natural log), "
                 "overlap vs two independent random sequences; N contributes 0",
        "rejection_rules": ["score <= 0", "any conflict with both qualities >= hq_min_q"],
        "quality_rounding": "round half up to nearest integer Phred",
    }
    return {
        "params": params,
        "inputs": {"r1": str(r1_path), "r2": str(r2_path)},
        "pairs": pair_results,
        "orphans": orphans,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------- 输出

def write_outputs(result: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = result["pairs"]

    # merged.fastq：仅位置可确定的配对
    with (outdir / "merged.fastq").open("w") as fh:
        for key in sorted(pairs):
            pr = pairs[key]
            if pr["status"] == STATUS_MERGED:
                fh.write(f"@{key}\n{pr['consensus_seq']}\n+\n{pr['consensus_qual']}\n")

    # unmerged.fastq：歧义/无重叠配对保留原始两端（原方向原名），孤儿保留原记录
    with (outdir / "unmerged.fastq").open("w") as fh:
        for key in sorted(pairs):
            pr = pairs[key]
            if pr["status"] in (STATUS_AMBIGUOUS, STATUS_NO_OVERLAP):
                for role in ("R1", "R2"):
                    # 原始记录需重取：result 中 pairs 仅存元数据，这里用诊断外的途径
                    rec = pr["_records"][role]
                    fh.write(f"@{rec.name}\n{rec.seq}\n+\n{rec.qual}\n")
        for rec in result["orphans"]:
            fh.write(f"@{rec.name}\n{rec.seq}\n+\n{rec.qual}\n")

    # pairs.tsv：逐配对状态与最佳/备选候选依据
    with (outdir / "pairs.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["key", "status", "reason", "n_candidates", "n_compatible",
                    "best_orientation", "best_d", "best_overlap", "best_score",
                    "best_rejected_reasons", "compatible_candidates"])
        for key in sorted(pairs):
            pr = pairs[key]
            best = pr["best"]
            w.writerow([
                key, pr["status"], pr["reason"], len(pr["candidates"]),
                len(pr["compatible"]),
                best.orientation if best else "", best.d if best else "",
                best.overlap if best else "",
                f"{best.score:.6f}" if best else "",
                "；".join(best.rejected) if best else "",
                ";".join(f"{c.orientation}:d={c.d},L={c.overlap},score={c.score:.3f}"
                         for c in pr["compatible"]),
            ])
        for rec in result["orphans"]:
            w.writerow([rec.key, STATUS_ORPHAN,
                        f"缺少另一端记录，原始记录保留于 unmerged.fastq",
                        "", "", "", "", "", "", "", ""])

    # candidates.tsv：全部候选评分表
    with (outdir / "candidates.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["key", "orientation", "d", "overlap", "n_informative",
                    "n_match", "n_mismatch", "n_hq_conflict", "score",
                    "compatible", "rejected_reasons"])
        for key in sorted(pairs):
            for c in pairs[key]["candidates"]:
                w.writerow([key, c.orientation, c.d, c.overlap, c.n_informative,
                            c.n_match, c.n_mismatch, len(c.hq_conflicts),
                            f"{c.score:.6f}", c.compatible, "；".join(c.rejected)])

    # provenance.tsv：merged.fastq 每个输出位置的来源
    with (outdir / "provenance.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["key", "pos", "consensus_base", "consensus_q", "source",
                    "fused", "r1_idx", "r1_base", "r1_q",
                    "r2_idx", "r2_base", "r2_q", "note"])
        for key in sorted(pairs):
            pr = pairs[key]
            if pr["status"] != STATUS_MERGED:
                continue
            for prov in pr["provenance"]:
                w.writerow([key, prov["pos"], prov["consensus_base"],
                            prov["consensus_q"], prov["source"], prov["fused"],
                            prov["r1_idx"] if prov["r1_idx"] is not None else "",
                            prov["r1_base"],
                            prov["r1_q"] if prov["r1_q"] is not None else "",
                            prov["r2_idx"] if prov["r2_idx"] is not None else "",
                            prov["r2_base"],
                            prov["r2_q"] if prov["r2_q"] is not None else "",
                            prov["note"]])

    # report.json：机器可读汇总
    report = {
        "params": result["params"],
        "inputs": result["inputs"],
        "summary": {s: sum(1 for pr in pairs.values() if pr["status"] == s)
                    for s in (STATUS_MERGED, STATUS_AMBIGUOUS, STATUS_NO_OVERLAP)},
        "orphans": [{"name": r.name, "key": r.key, "file": r.file, "line": r.line}
                    for r in result["orphans"]],
        "diagnostics": result["diagnostics"],
        "pairs": {},
    }
    for key in sorted(pairs):
        pr = pairs[key]
        report["pairs"][key] = {
            "status": pr["status"], "reason": pr["reason"],
            "r1": pr["r1"], "r2": pr["r2"],
            "best_candidate": pr["best"].to_dict(full=True) if pr["best"] else None,
            "compatible_candidates": [c.to_dict(full=True) for c in pr["compatible"]],
            "candidates": [c.to_dict() for c in pr["candidates"]],
            "consensus_seq": pr["consensus_seq"],
            "consensus_qual": pr["consensus_qual"],
            "provenance": pr["provenance"],
        }
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merge",
        description="本地双端短读段合并：Phred+33 / ACGTN / 无indel末端重叠")
    ap.add_argument("--r1", default="fixtures/reads_R1.fastq", help="R1 FASTQ")
    ap.add_argument("--r2", default="fixtures/reads_R2.fastq", help="R2 FASTQ")
    ap.add_argument("--plan", default="fixtures/plan.json", help="参数 plan JSON")
    ap.add_argument("--outdir", default="out", help="输出目录")
    ap.add_argument("--hq-min-q", type=int, default=HQ_MIN_Q,
                    help="高质量矛盾阈值（双端 Q 均 >= 该值且碱基不同则拒绝候选）")
    args = ap.parse_args(argv)

    plan = load_plan(Path(args.plan))
    result = run_pipeline(Path(args.r1), Path(args.r2), plan, args.hq_min_q)
    write_outputs(result, Path(args.outdir))

    summary = {}
    for pr in result["pairs"].values():
        summary[pr["status"]] = summary.get(pr["status"], 0) + 1
    print(f"配对 {len(result['pairs'])} 个: " +
          ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) +
          f"；孤儿 {len(result['orphans'])} 个；诊断 {len(result['diagnostics'])} 条")
    for key in sorted(result["pairs"]):
        pr = result["pairs"][key]
        print(f"  {key}: {pr['status']} — {pr['reason']}")
    print(f"输出目录: {args.outdir}/ (merged.fastq, unmerged.fastq, pairs.tsv, "
          f"candidates.tsv, provenance.tsv, report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
