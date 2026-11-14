"""精确求解器：子集动态规划 + 最优路径上的探针歧义枚举。

最优性依据（详见 METHOD.md）：
- 状态 f[S]：恰使用谓词集合 S（位掩码）的有序规则序列，在「未被任何
  S 中规则遮蔽」的案例上所能达到的最小累计错送代价 + 每规则维护代价。
- 转移 f[S∪{i}] = min_{i∉S, q} f[S] + rule_cost(i,q, ~union(S)) + penalty，
  rule_cost 是规则 (i→q) 在其首次命中案例上的加权错送代价。首次命中只
  依赖「此前用了哪些谓词」而与顺序无关，故最优子结构成立；DP 遍历全部
  谓词子集 × 队列，不做单条规则贪心。
- 空规则集 f[0]=0 是合法状态；默认队列 d 的错送单独计入 default_cost[d]，
  总目标 = f[S] + default_cost[d]。
- 探针歧义只在目标值等于全局最小值 V* 的路径（同值全局最优方案）上枚举，
  可行性由「受目标集合约束的后缀 DP」精确判定，不漏不错。
所有枚举次序均以谓词 id / 队列名字典序为准，与输入行顺序无关。
"""

from __future__ import annotations

from .data import Case, CostMatrix, Plan, Predicate
from .predicates import evaluate

INF = float("inf")
TOL = 1e-9


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class ExactSolver:
    def __init__(
        self,
        predicates: list[Predicate],
        cases: list[Case],
        costs: CostMatrix,
        plan: Plan,
        match_masks: list[int],
    ):
        self.preds = predicates
        self.cases = cases
        self.costs = costs
        self.plan = plan
        self.match = match_masks

        self.n = len(predicates)
        self.m = len(cases)
        self.all_bits = (1 << self.m) - 1
        self.queues = costs.queues
        self.k = min(plan.max_rules, self.n)
        self.penalty = plan.rule_penalty

        self._weights = [c.weight for c in cases]
        self._targets = [c.target for c in cases]

        # 稳定枚举顺序：谓词 id 字典序（不依赖输入行顺序）
        self.order = sorted(range(self.n), key=lambda i: predicates[i].id)
        self.full = 1 << self.n

        self._union = self._precompute_union()
        # 默认错送只发生在「未被任何规则命中」的案例上，故与谓词集合 S 有关：
        # default_cost(d, S) 按案例掩码 ~union(S) 计算并记忆化。
        self._default_cache: dict = {}
        # rule_cost(i,q,available_case_mask) 按需记忆化：
        # 可达的案例掩码只能是若干谓词匹配集的并/交，数量远小于 2^m
        self._rule_cost_cache: dict = {}

        self.f: list[float] = []
        self._h_fixed_memo: dict = {}

    # ------------------------------------------------------------------ 预处理
    def _precompute_union(self) -> list[int]:
        union = [0] * self.full
        for mask in range(1, self.full):
            low = mask & -mask
            i = low.bit_length() - 1
            union[mask] = union[mask ^ low] | self.match[i]
        return union

    def default_cost(self, d: str, used_set: int) -> float:
        """默认队列 d 在未被 used_set 中任何谓词命中的案例上的加权错送。"""
        key = (d, used_set)
        cached = self._default_cache.get(key)
        if cached is not None:
            return cached
        remaining = self.all_bits & ~self._union[used_set]
        w, tgt = self._weights, self._targets
        total = 0.0
        for j in bits(remaining):
            total += w[j] * self.costs.cost[tgt[j]][d]
        self._default_cache[key] = total
        return total

    def rule_cost(self, i: int, q: str, available: int) -> float:
        """规则 (i→q) 在当前可用案例掩码上的加权错送代价（首次命中部分）。"""
        key = (i, q, available)
        cached = self._rule_cost_cache.get(key)
        if cached is not None:
            return cached
        first_hit = self.match[i] & available
        matrix_q = self.costs.cost
        w, tgt = self._weights, self._targets
        total = 0.0
        for j in bits(first_hit):
            total += w[j] * matrix_q[tgt[j]][q]
        self._rule_cost_cache[key] = total
        return total

    # ------------------------------------------------------------------ 前向子集 DP
    def solve_forward(self) -> None:
        f = [INF] * self.full
        f[0] = 0
        for s in range(self.full):
            base = f[s]
            if base == INF or s.bit_count() >= self.k:
                continue
            available = self.all_bits & ~self._union[s]
            for i in self.order:
                bit = 1 << i
                if s & bit:
                    continue
                ns = s | bit
                for q in self.queues:
                    cand = base + self.rule_cost(i, q, available) + self.penalty
                    if cand < f[ns]:
                        f[ns] = cand
        self.f = f

    # ------------------------------------------------- 受约束后缀 DP（可行性 oracle）
    def h_fixed(self, used: int, available: int, target_set: int, slots: int) -> float:
        """必须恰好用完 target_set\\used 中全部谓词、且至多再用 slots 条规则时，
        最小后缀规则侧增量（含维护代价）；槽位不足返回 INF。"""
        need = (target_set & ~used).bit_count()
        if need > slots:
            return INF
        if need == 0:
            return 0.0
        key = (used, available, target_set, slots)
        cached = self._h_fixed_memo.get(key)
        if cached is not None:
            return cached
        best = INF
        for i in self.order:
            bit = 1 << i
            if not (target_set & bit) or (used & bit):
                continue
            na = available & ~self.match[i]
            for q in self.queues:
                inc = self.rule_cost(i, q, available) + self.penalty
                cand = inc + self.h_fixed(used | bit, na, target_set, slots - 1)
                if cand < best:
                    best = cand
        self._h_fixed_memo[key] = best
        return best

    # ------------------------------------------------------------------ 最优终端
    def terminals(self) -> list[dict]:
        """返回全部同值最优终端 (谓词集合 S, 默认队列 d)。"""
        scored = [
            {"set": s, "default": d,
             "value": self.f[s] + self.default_cost(d, s)}
            for s in range(self.full)
            if self.f[s] != INF
            for d in self.plan.allow_default
        ]
        best = min(t["value"] for t in scored)
        return [t for t in scored if abs(t["value"] - best) <= TOL]

    def optimum_value(self, terminals: list[dict] | None = None) -> float:
        terms = terminals or self.terminals()
        return min(t["value"] for t in terms)

    # ------------------------------------------------------------------ 规范主方案
    def _terminal_key(self, term: dict) -> tuple:
        ids = tuple(sorted(self.preds[i].id for i in bits(term["set"])))
        return (len(ids), ids, term["default"])

    def canonical_terminal(self, terminals: list[dict]) -> dict:
        return min(terminals, key=self._terminal_key)

    def canonical_plan(self, term: dict) -> dict:
        """贪心重建字典序最小的有序规则序列。

        每一步选满足「受约束后缀恰好补齐 f[S*]」的最小 (谓词id, 队列)；
        可行性由 h_fixed 精确判定，故重建序列必达到该终端的最优值。
        """
        target_set, d = term["set"], term["default"]
        used, available, prefix, slots = 0, self.all_bits, 0.0, self.k
        chosen: list[tuple] = []
        while used != target_set:
            candidates = []
            for i in self.order:
                bit = 1 << i
                if not (target_set & bit) or (used & bit):
                    continue
                na = available & ~self.match[i]
                for q in self.queues:
                    inc = self.rule_cost(i, q, available) + self.penalty
                    tail = self.h_fixed(used | bit, na, target_set, slots - 1)
                    if abs(prefix + inc + tail - self.f[target_set]) <= TOL:
                        candidates.append((self.preds[i].id, q, i, inc, na))
            if not candidates:  # 防御性断言：正常数据下不可达
                raise RuntimeError("规范方案重建失败：DP 状态不一致")
            _pid, q, i, inc, na = min(candidates, key=lambda x: (x[0], x[1]))
            chosen.append((i, q))
            used |= 1 << i
            available = na
            prefix += inc
            slots -= 1
        return {
            "rules": [
                {"predicate": self.preds[i].to_dict(), "queue": q} for i, q in chosen
            ],
            "default_queue": d,
        }

    # ------------------------------------------------------------------ 探针歧义
    def probe_match_masks(self, probe_rows: list[dict]) -> list[int]:
        out = []
        for row in probe_rows:
            mask = 0
            for i, pred in enumerate(self.preds):
                if evaluate(pred, row["fields"]):
                    mask |= 1 << i
            out.append(mask)
        return out

    def probe_ambiguity(
        self, probe_rows: list[dict], terminals: list[dict], v_star: float
    ) -> list[dict]:
        """逐探针求「全部同值全局最优方案」上的可达队列与两份分流不同的证据。

        同值最优方案除基础终端的有效规则外，还可能包含**冗余规则**：其匹配
        案例已被前面规则全部遮蔽，训练侧增量只有维护代价 penalty，但仍可能
        命中探针并改变其分流。对每个基础终端 (S*, d) 单独做带槽位的记忆化
        DFS；记忆值保存完整规则路径，探针队列在汇总时由路径逐条重放决定：
        - 有效边 i∈S*：要求重放后的当前路径代价 + 边增量 + 受约束后缀最优
          恰为 f[S*]，即始终处在该终端的最优轨道上；
        - 冗余边 i∉S*：当前训练侧零捕获，增量恰为 penalty，且剩余槽位仍够
          补齐所有未用有效谓词。
        每条到达目标状态的路径，训练目标值都严格等于 V*，verify 再独立复核。
        """
        probe_masks = self.probe_match_masks(probe_rows)
        term_by_default: dict = {}
        for t in terminals:
            term_by_default.setdefault(t["default"], []).append(t["set"])
        ordered_terminals = [
            (d, s_star)
            for d in sorted(term_by_default)
            for s_star in sorted(
                term_by_default[d],
                key=lambda s: (
                    s.bit_count(),
                    tuple(sorted(self.preds[i].id for i in bits(s))),
                ),
            )
        ]

        reports = []
        for row, pmask in zip(probe_rows, probe_masks):
            reachable: dict = {}  # queue -> 一份完整最优方案
            for d, s_star in ordered_terminals:
                memo: dict = {}
                for suffix in self._probe_dfs(
                    used=0,
                    available=self.all_bits,
                    slots=self.k,
                    target_set=s_star,
                    path=[],
                    memo=memo,
                ):
                    queue = self._route_on_path(suffix, pmask, d)
                    reachable.setdefault(
                        queue, {"rules": suffix, "default": d}
                    )
            witnesses = [
                {
                    "queue": q,
                    "plan": {
                        "rules": [
                            {"predicate": self.preds[i].to_dict(), "queue": qq}
                            for i, qq in reachable[q]["rules"]
                        ],
                        "default_queue": reachable[q]["default"],
                    },
                }
                for q in sorted(reachable)
            ]
            reports.append(
                {
                    "probe_id": row["id"],
                    "matched_predicate_ids": sorted(
                        self.preds[i].id for i in bits(pmask)
                    ),
                    "reachable_queues": sorted(reachable),
                    "determined": len(reachable) == 1,
                    # 有歧义时给出可证明不同分流的方案；确定时不需要证据清单
                    "witnesses": witnesses if len(reachable) > 1 else [],
                }
            )
        return reports

    def _probe_dfs(
        self,
        used: int,
        available: int,
        slots: int,
        target_set: int,
        path: list,
        memo: dict,
    ) -> list:
        """枚举该基础终端下所有同值最优的完整规则路径（含合规冗余规则）。

        记忆键 (used, available, slots) 对当前终端唯一；记忆值是从该状态出发
        的后缀路径列表（已含本状态之后的全部规则），与到达该状态的路径无关。
        """
        key = (used, available, slots)
        if key in memo:
            return memo[key]

        completions: list = []
        if used == target_set:
            completions.append([])  # 空后缀：当前 path 已构成完整方案

        target_remaining = target_set & ~used
        for i in self.order:
            bit = 1 << i
            if used & bit or slots <= 0:
                continue
            na = available & ~self.match[i]
            for q in self.queues:
                feasible = False
                if target_set & bit:
                    inc = self.rule_cost(i, q, available) + self.penalty
                    tail = self.h_fixed(used | bit, na, target_set, slots - 1)
                    # 重放当前完整路径（含冗余规则的遮蔽与 penalty）+ 本边 + 最优后缀
                    if abs(
                        self._path_cost(path) + inc + tail - self.f[target_set]
                    ) <= TOL:
                        feasible = True
                elif (self.match[i] & available) == 0:
                    # 冗余边：训练侧零捕获；剩余槽位仍够补齐有效谓词
                    if target_remaining.bit_count() <= slots - 1:
                        feasible = True
                if not feasible:
                    continue
                for tail_path in self._probe_dfs(
                    used | bit, na, slots - 1, target_set,
                    path + [(i, q)], memo,
                ):
                    completions.append([(i, q)] + tail_path)

        uniq, seen_sig = [], set()
        for comp in completions:
            sig = tuple(comp)
            if sig not in seen_sig:
                seen_sig.add(sig)
                uniq.append(comp)
        memo[key] = uniq
        return uniq

    def _route_on_path(self, path: list, probe_mask: int, default_queue: str) -> str:
        """按有序规则路径对单个探针（谓词命中掩码）决定首次命中队列。"""
        for i, q in path:
            if probe_mask & (1 << i):
                return q
        return default_queue

    def _path_cost(self, path: list) -> float:
        """从头重放一条规则路径的规则侧代价（错送 + 维护）。"""
        available, total = self.all_bits, 0.0
        for i, q in path:
            total += self.rule_cost(i, q, available) + self.penalty
            available &= ~self.match[i]
        return total
