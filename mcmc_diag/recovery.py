"""按 manifest 从分片 CSV 恢复链。

恢复语义（对所有参数一致）：
1. 文件顺序无意义：片段按 (chain_id, start_iteration, end_iteration, path) 排序处理，
   链按 chain_id 排序，保证 manifest 中条目顺序不影响结果。
2. 同一 chain_id 的片段合并为一条链；片段不会被当作独立链，链之间也不会拼接。
3. 元信息矛盾阻断该链：同一链各片段的 warmup_end / thin 必须一致；
   manifest 声明的 [start_iteration, end_iteration]（步长 thin）必须与文件实际
   iteration 集合完全一致。
4. 合并时按 iteration 对齐：完全相同的重复观察（iteration 与全部参数值逐一相等）
   只计一次并记录去重数量；iteration 相同但任一参数值不同即冲突重复，阻断该链。
5. 合并后 iteration 必须按 thin 连续，链内缺段阻断该链。
6. 去除各自 warmup：iteration <= warmup_end 的样本丢弃。
7. 多链诊断使用公开的共同保留长度 L = 各有效链保留样本数的最小值；
   每条链保留 warmup 后最早的 L 个样本，尾部多余样本丢弃并逐链记录数量。
   不允许为了结果好看挑选片段：截断规则固定为“保留最早、丢弃尾部”。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHAIN_VALID = "valid"
CHAIN_BLOCKED = "blocked"

ITERATION_COLUMN = "iteration"


@dataclass
class FragmentRecord:
    """单个片段文件的读取与核查记录。"""

    chain_id: str
    path: str
    declared_start: int
    declared_end: int
    warmup_end: int
    thin: int
    n_rows: int = 0
    first_iteration: int | None = None
    last_iteration: int | None = None
    duplicates_removed: int = 0
    status: str = "ok"
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "path": self.path,
            "declared_start_iteration": self.declared_start,
            "declared_end_iteration": self.declared_end,
            "warmup_end": self.warmup_end,
            "thin": self.thin,
            "n_rows": self.n_rows,
            "first_iteration": self.first_iteration,
            "last_iteration": self.last_iteration,
            "duplicates_removed": self.duplicates_removed,
            "status": self.status,
            "problems": list(self.problems),
        }


@dataclass
class ChainRecord:
    """一条链的恢复记录（含被阻断的链）。"""

    chain_id: str
    status: str
    problems: list[str]
    warmup_end: int | None
    thin: int | None
    fragments: list[FragmentRecord]
    n_rows_loaded: int = 0
    n_duplicates_removed: int = 0
    n_warmup_removed: int = 0
    n_retained: int = 0
    retained_first_iteration: int | None = None
    retained_last_iteration: int | None = None
    common_length: int | None = None
    n_discarded_tail: int | None = None
    # 以下仅对 valid 链填充：warmup 后、共同长度截断前的完整保留样本
    iterations: np.ndarray | None = None
    values: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "status": self.status,
            "problems": list(self.problems),
            "warmup_end": self.warmup_end,
            "thin": self.thin,
            "fragments": [f.to_dict() for f in self.fragments],
            "n_rows_loaded": self.n_rows_loaded,
            "n_duplicates_removed": self.n_duplicates_removed,
            "n_warmup_removed": self.n_warmup_removed,
            "n_retained_after_warmup": self.n_retained,
            "retained_first_iteration": self.retained_first_iteration,
            "retained_last_iteration": self.retained_last_iteration,
            "common_length_used": self.common_length,
            "n_discarded_tail": self.n_discarded_tail,
        }


@dataclass
class RecoveredData:
    """恢复结果：全部链记录 + 多链诊断实际使用的等长样本。"""

    parameters: list[str]
    chains: list[ChainRecord]
    valid_chain_ids: list[str]
    common_length: int | None
    used_iterations: dict[str, np.ndarray]  # chain_id -> 使用的 iteration（长度 common_length）
    used_values: dict[str, dict[str, np.ndarray]]  # chain_id -> param -> 等长样本
    decision_limits: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def used_matrix(self, parameter: str) -> np.ndarray:
        """返回 (n_valid_chains, common_length) 矩阵，链顺序固定为 valid_chain_ids。"""
        return np.array([self.used_values[c][parameter] for c in self.valid_chain_ids])

    def to_record(self) -> dict:
        return {
            "parameters": list(self.parameters),
            "valid_chain_ids": list(self.valid_chain_ids),
            "common_length": self.common_length,
            "decision_limits": dict(self.decision_limits),
            "notes": list(self.notes),
            "chains": [c.to_dict() for c in self.chains],
        }


class ManifestError(ValueError):
    """manifest 本身不合法（区别于单链可恢复问题）。"""


def load_manifest(manifest_path: str | Path) -> tuple[dict, Path]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for key in ("files", "parameters"):
        if key not in manifest:
            raise ManifestError(f"manifest 缺少必需字段: {key}")
    return manifest, path.parent


def _read_fragment_csv(csv_path: Path, parameters: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """读取一个片段 CSV，返回 (iterations, {param: values})。

    解析失败（缺列、行无法解析为数值）抛出 ValueError，由调用方记为该链问题。
    """
    iterations: list[int] = []
    columns: dict[str, list[float]] = {p: [] for p in parameters}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in [ITERATION_COLUMN, *parameters] if c not in header]
        if missing:
            raise ValueError(f"缺少列: {missing}")
        for lineno, row in enumerate(reader, start=2):
            try:
                it_text = (row[ITERATION_COLUMN] or "").strip()
                it = int(it_text)
                if float(it_text) != it:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"第 {lineno} 行 iteration 无法解析为整数: {row[ITERATION_COLUMN]!r}")
            vals = []
            for p in parameters:
                try:
                    vals.append(float(row[p]))
                except (TypeError, ValueError):
                    raise ValueError(f"第 {lineno} 行参数 {p} 无法解析为数值: {row[p]!r}")
            iterations.append(it)
            for p, v in zip(parameters, vals):
                columns[p].append(v)
    if not iterations:
        raise ValueError("文件无数据行")
    return np.array(iterations, dtype=np.int64), {p: np.array(v, dtype=float) for p, v in columns.items()}


def _expected_iterations(start: int, end: int, thin: int) -> np.ndarray:
    return np.arange(start, end + 1, thin, dtype=np.int64)


def recover_chains(
    manifest_path: str | Path,
    extra_entries: list[dict] | None = None,
) -> RecoveredData:
    """按 manifest（可附加额外片段条目）恢复全部链。

    extra_entries: 与 manifest["files"] 同构的条目列表（如 cases/add_fragment.json
    中的 add_to_manifest）。路径相对 manifest 所在目录解析。
    """
    manifest, base_dir = load_manifest(manifest_path)
    parameters = list(manifest["parameters"])
    entries = list(manifest["files"]) + list(extra_entries or [])
    limits = {k: float(v) for k, v in manifest.get("decision_limits", {}).items()}

    # 按链分组；排序保证文件顺序不影响结果
    grouped: dict[str, list[dict]] = {}
    for e in entries:
        for key in ("chain_id", "path", "start_iteration", "end_iteration", "warmup_end", "thin"):
            if key not in e:
                raise ManifestError(f"manifest 条目缺少字段 {key}: {e}")
        grouped.setdefault(str(e["chain_id"]), []).append(e)

    chains: list[ChainRecord] = []
    for chain_id in sorted(grouped):
        chains.append(
            _recover_one_chain(chain_id, sorted(grouped[chain_id], key=lambda e: (e["start_iteration"], e["end_iteration"], e["path"])), parameters, base_dir)
        )

    valid = [c for c in chains if c.status == CHAIN_VALID]
    notes: list[str] = []
    used_iterations: dict[str, np.ndarray] = {}
    used_values: dict[str, dict[str, np.ndarray]] = {}
    common_length: int | None = None
    if valid:
        common_length = min(c.n_retained for c in valid)
        for c in valid:
            c.common_length = common_length
            c.n_discarded_tail = c.n_retained - common_length
            used_iterations[c.chain_id] = c.iterations[:common_length]
            used_values[c.chain_id] = {p: v[:common_length] for p, v in c.values.items()}
        blocked = [c.chain_id for c in chains if c.status == CHAIN_BLOCKED]
        if blocked:
            notes.append(f"链 {blocked} 被阻断，多链诊断仅使用有效链 {[c.chain_id for c in valid]}。")
        if any(c.n_discarded_tail for c in valid):
            notes.append(
                "共同保留长度 L=%d；各链丢弃尾部样本数: %s。"
                % (common_length, {c.chain_id: c.n_discarded_tail for c in valid})
            )
    else:
        notes.append("没有任何有效链，无法执行多链诊断。")

    return RecoveredData(
        parameters=parameters,
        chains=chains,
        valid_chain_ids=[c.chain_id for c in valid],
        common_length=common_length,
        used_iterations=used_iterations,
        used_values=used_values,
        decision_limits=limits,
        notes=notes,
    )


def _recover_one_chain(chain_id: str, entries: list[dict], parameters: list[str], base_dir: Path) -> ChainRecord:
    problems: list[str] = []
    fragments: list[FragmentRecord] = []

    warmup_ends = {int(e["warmup_end"]) for e in entries}
    thins = {int(e["thin"]) for e in entries}
    warmup_end = warmup_ends.pop() if len(warmup_ends) == 1 else None
    thin = thins.pop() if len(thins) == 1 else None
    meta_ok = True
    if warmup_end is None:
        problems.append(f"元信息矛盾: 同一链各片段 warmup_end 不一致 {sorted({int(e['warmup_end']) for e in entries})}")
        meta_ok = False
    if thin is None:
        problems.append(f"元信息矛盾: 同一链各片段 thin 不一致 {sorted({int(e['thin']) for e in entries})}")
        meta_ok = False
    if thin is not None and thin < 1:
        problems.append(f"元信息矛盾: thin={thin} 非法")
        meta_ok = False

    # 读取并核查每个片段
    loaded: list[tuple[FragmentRecord, np.ndarray, dict[str, np.ndarray]]] = []
    for e in entries:
        rec = FragmentRecord(
            chain_id=chain_id,
            path=str(e["path"]),
            declared_start=int(e["start_iteration"]),
            declared_end=int(e["end_iteration"]),
            warmup_end=int(e["warmup_end"]),
            thin=int(e["thin"]),
        )
        fragments.append(rec)
        try:
            its, cols = _read_fragment_csv(base_dir / rec.path, parameters)
        except (OSError, ValueError) as exc:
            rec.status = "error"
            rec.problems.append(f"读取失败: {exc}")
            problems.append(f"片段 {rec.path}: 读取失败: {exc}")
            continue
        rec.n_rows = len(its)
        rec.first_iteration = int(its.min())
        rec.last_iteration = int(its.max())
        # manifest 声明区间必须与文件实际 iteration 集合一致（完全重复的行在合并阶段去重）
        expected = _expected_iterations(rec.declared_start, rec.declared_end, rec.thin)
        if not np.array_equal(np.unique(its), expected):
            rec.status = "error"
            rec.problems.append(
                f"声明区间 [{rec.declared_start},{rec.declared_end}] thin={rec.thin} 与文件实际 iteration 集合不一致"
            )
            problems.append(f"片段 {rec.path}: 声明区间与文件内容矛盾")
        loaded.append((rec, its, cols))

    # 合并：iteration 对齐去重 / 冲突检测（在任何读取失败时仍尝试，便于完整记录）
    merged: dict[int, tuple[float, ...]] = {}
    conflicts: list[int] = []
    n_rows_loaded = 0
    n_duplicates = 0
    for rec, its, cols in loaded:
        for idx, it in enumerate(its):
            n_rows_loaded += 1
            row = tuple(float(cols[p][idx]) for p in parameters)
            key = int(it)
            if key in merged:
                if merged[key] == row:  # 完全相同的重复观察只计一次
                    rec.duplicates_removed += 1
                    n_duplicates += 1
                else:
                    conflicts.append(key)
            else:
                merged[key] = row
    for rec, its, cols in loaded:
        if rec.duplicates_removed:
            rec.problems.append(f"与其他片段完全重复 {rec.duplicates_removed} 行，已去重（仅计一次）")
    if conflicts:
        uniq = sorted(set(conflicts))
        problems.append(f"冲突重复: iteration {uniq} 在片段间取值不一致，阻断该链诊断")

    # 链内缺段检查
    gap_missing: list[int] = []
    if merged and meta_ok:
        ordered = np.array(sorted(merged), dtype=np.int64)
        expected_full = _expected_iterations(int(ordered[0]), int(ordered[-1]), thin)
        if len(ordered) != len(expected_full):
            missing = sorted(set(expected_full.tolist()) - set(ordered.tolist()))
            gap_missing = missing
            problems.append(f"链内缺段: 缺少 iteration {missing[:10]}{'...' if len(missing) > 10 else ''}，阻断该链诊断")

    record = ChainRecord(
        chain_id=chain_id,
        status=CHAIN_VALID,
        problems=problems,
        warmup_end=warmup_end,
        thin=thin,
        fragments=fragments,
        n_rows_loaded=n_rows_loaded,
        n_duplicates_removed=n_duplicates,
    )

    if problems:
        record.status = CHAIN_BLOCKED
        return record

    # 排序并去除 warmup（iteration <= warmup_end）
    ordered = np.array(sorted(merged), dtype=np.int64)
    mat = np.array([merged[int(it)] for it in ordered], dtype=float)
    keep = ordered > warmup_end
    record.n_warmup_removed = int((~keep).sum())
    kept_it = ordered[keep]
    kept = mat[keep]
    record.n_retained = len(kept_it)
    if record.n_retained == 0:
        record.status = CHAIN_BLOCKED
        record.problems.append(f"去除 warmup（iteration <= {warmup_end}）后无剩余样本，阻断该链诊断")
        return record
    record.retained_first_iteration = int(kept_it[0])
    record.retained_last_iteration = int(kept_it[-1])
    record.iterations = kept_it
    record.values = {p: kept[:, j] for j, p in enumerate(parameters)}
    return record
