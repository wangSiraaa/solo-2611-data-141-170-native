"""抽样资格审核：漏序、重复、改序、未知件号等。

资格与统计证据是两部分：资格失败时保留全部诊断，但不得继续出具有效审计证书。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .io import ObsRow, Plan
from .order import derive_order


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def to_json(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class Eligibility:
    ok: bool
    checks: list[Check]

    @property
    def diagnostics(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok]

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.to_json() for c in self.checks],
            "diagnostics": self.diagnostics,
        }


def check_eligibility(plan: Plan, lot_ids: list[str], obs_rows: list[ObsRow]) -> Eligibility:
    checks: list[Check] = []

    # 1. 名单资格：件数与 plan 一致、件号唯一（否则抽查算法本身无定义）。
    unique_ids = sorted(set(lot_ids))
    dup_in_lot = sorted({i for i in unique_ids if lot_ids.count(i) > 1})
    roster_ok = len(lot_ids) == plan.lot_size == len(unique_ids)
    detail = (
        f"名单 {len(lot_ids)} 件，唯一件号 {len(unique_ids)} 个，"
        f"plan.lot_size={plan.lot_size}"
    )
    if dup_in_lot:
        detail += f"；名单内重复件号: {dup_in_lot[:5]}"
    checks.append(Check("lot_roster_matches_plan", roster_ok, detail))

    order = derive_order(lot_ids, plan.seed) if roster_ok else None
    if order is not None:
        order_digest = hashlib.sha256(" ".join(order).encode("utf-8")).hexdigest()
        checks.append(
            Check(
                "draw_order_reproducible",
                True,
                f"按 plan 算法复现抽查顺序 {len(order)} 件，sha256={order_digest}",
            )
        )

    # 2. 记录格式：draw_no 为正整数、defect 为 0/1、item_id 非空。
    bad: list[str] = []
    for r in obs_rows:
        if not r.draw_no_raw.isdigit():
            bad.append(f"第{r.line}行 draw_no={r.draw_no_raw!r} 非正整数")
        if r.defect_raw not in ("0", "1"):
            bad.append(f"第{r.line}行 defect={r.defect_raw!r} 非 0/1")
        if not r.item_id:
            bad.append(f"第{r.line}行 item_id 为空")
    fmt_ok = not bad
    checks.append(
        Check(
            "record_format",
            fmt_ok,
            "; ".join(bad[:10]) if bad else f"{len(obs_rows)} 行记录格式合法",
        )
    )
    draw_nos = [int(r.draw_no_raw) for r in obs_rows] if fmt_ok else None

    # 3. 抽号连续无漏序：draw_no 必须恰为 1..m。
    if draw_nos is None:
        checks.append(
            Check("draw_sequence_contiguous", False, "记录格式不合法，无法检查抽号序列")
        )
    else:
        expected = list(range(1, len(obs_rows) + 1))
        if draw_nos == expected:
            checks.append(
                Check(
                    "draw_sequence_contiguous",
                    True,
                    f"draw_no 恰为 1..{len(obs_rows)} 的连续前缀",
                )
            )
        else:
            got = set(draw_nos)
            missing = [i for i in expected if i not in got]
            dup_no = sorted({d for d in got if draw_nos.count(d) > 1})
            checks.append(
                Check(
                    "draw_sequence_contiguous",
                    False,
                    f"抽号序列不是 1..{len(obs_rows)} 的连续前缀；"
                    f"缺失: {missing[:10]}；重复抽号: {dup_no[:10]}",
                )
            )

    # 4. 观察数不超过批量。
    m = len(obs_rows)
    checks.append(
        Check(
            "observation_count_within_lot",
            m <= plan.lot_size,
            f"观察 {m} 条，批量 {plan.lot_size}",
        )
    )

    # 5. 件号已知。
    lot_set = set(lot_ids)
    unknown = [(r.line, r.item_id) for r in obs_rows if r.item_id not in lot_set]
    checks.append(
        Check(
            "items_known",
            not unknown,
            "全部件号在名单内"
            if not unknown
            else f"未知件号(行号,件号): {unknown[:10]}",
        )
    )

    # 6. 件号无重复（无放回抽查，每件至多出现一次）。
    first_seen: dict[str, int] = {}
    dups: list[tuple[str, int, int]] = []
    for r in obs_rows:
        if r.item_id in first_seen:
            dups.append((r.item_id, first_seen[r.item_id], r.line))
        else:
            first_seen[r.item_id] = r.line
    checks.append(
        Check(
            "items_unique",
            not dups,
            "无重复件号"
            if not dups
            else f"重复件号(件号,首次行,再次行): {dups[:10]}",
        )
    )

    # 7. 与 plan 抽查顺序的连续前缀一致（改序/张冠李戴在此暴露）。
    if order is None:
        checks.append(
            Check(
                "prefix_matches_plan_order",
                False,
                "名单资格失败，无法复现抽查顺序，跳过前缀核对",
            )
        )
    else:
        mism: list[str] = []
        for i, r in enumerate(obs_rows):
            if i >= len(order):
                break
            want = order[i]
            if r.item_id != want:
                if r.item_id in order:
                    note = f"（该件在 plan 顺序第 {order.index(r.item_id) + 1} 位）"
                elif r.item_id not in lot_set:
                    note = "（未知件号）"
                else:
                    note = ""
                mism.append(f"抽号{i + 1}: 应为 {want}，实为 {r.item_id}{note}")
        prefix_ok = not mism and m <= len(order)
        detail = (
            f"观察序列与 plan 抽查顺序的前 {m} 位一致"
            if prefix_ok
            else "; ".join(mism[:10]) + (f" …共 {len(mism)} 处不符" if len(mism) > 10 else "")
        )
        checks.append(Check("prefix_matches_plan_order", prefix_ok, detail))

    return Eligibility(ok=all(c.ok for c in checks), checks=checks)
