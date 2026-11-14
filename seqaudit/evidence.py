"""顺序证据：混合似然比检验鞅（有限总体、无放回、精确有理数计算）。

定义
----
整批 N 件，缺陷总数 D 固定未知。复合零假设 H0: D >= d0（d0 为 plan 的拒收阈值）。
按简单随机无放回顺序揭示 x_1..x_t（1=缺陷），k_t = 累计缺陷数。

证据过程为"混合备择 vs 边界零假设"的似然比：

    M_t = sum_{d1=0}^{d0-1} w(d1) * P_{D=d1}(x_1..x_t) / P_{D=d0}(x_1..x_t)

其中 P_D(具体有序模式, 含 k 个缺陷) = C(N-t, D-k) / C(N, D)，故

    M_t = sum_{d1<d0} w(d1) * C(N-t, d1-k) * C(N, d0) / ( C(N, d1) * C(N-t, d0-k) )。

性质（证明见 docs/METHOD.md）：
  * M_0 = 1，M_t >= 0；
  * 在边界零假设 D=d0 下 M_t 是检验鞅（条件期望恰为 1）；
  * 由超几何似然比关于 k 的单调性，M_t 在任意 D >= d0 下是非负上鞅；
  * Ville 不等式：对任意 D >= d0 的批次与任意（可依赖已揭示结果的）停止规则，
    P(存在 t 使 M_t >= 1/alpha) <= alpha。接受规则即"M_t >= 1/alpha"。

似然项无法定义时的统计语义（不是数值掩盖）：
  * k > d0：已观察到的缺陷数本身证明 D >= k > d0-1，零假设必然为真
    => 证据 = 0（此后永远不得接受）；
  * d0 - k > N - t：剩余未查件即使全部缺陷也凑不到 d0，零假设被逻辑否证
    => 证据 = +inf（接受是无风险的逻辑结论，只可能在接近全量检查时出现）。

数值实现：math.comb 精确整数 + Fraction 精确有理数；alpha 由字符串精确解析；
不用浮点参与判定（浮点仅供展示）。
"""
from __future__ import annotations

from fractions import Fraction
from math import comb, inf

#: 证据值类型：Fraction，或 +inf（零假设被逻辑否证）。
INF = inf


def mixture_weights(d0: int) -> tuple[Fraction, ...]:
    """冻结的设计参数：备择混合权重，均匀分布在 d1 = 0..d0-1 上。

    权重在读取任何缺陷结果之前冻结（只依赖 plan 的 d0），
    更换权重只影响检验的效，不影响风险上限的有效性。
    """
    return tuple(Fraction(1, d0) for _ in range(d0))


def _c(n: int, k: int) -> int:
    """组合数，越界按 0 处理（对应"该缺陷总数下此观察不可能"）。"""
    return comb(n, k) if 0 <= k <= n else 0


def evidence_value(
    n: int, d0: int, t: int, k: int, weights: tuple[Fraction, ...] | None = None
) -> Fraction | float:
    """第 t 抽、累计 k 个缺陷后的证据 M_t（组合数形式，精确）。"""
    if weights is None:
        weights = mixture_weights(d0)
    remaining = n - t
    if d0 - k > remaining:
        return INF  # 零假设被逻辑否证：剩余件全缺陷也到不了 d0
    if k > d0:
        return Fraction(0)  # 零假设必然为真：已见缺陷数超过边界 d0
    den = _c(remaining, d0 - k)  # 上述分支保证 den > 0
    cn_d0 = comb(n, d0)
    total = Fraction(0)
    for d1, w in enumerate(weights):
        if not w:
            continue
        num = _c(remaining, d1 - k)
        if num:
            total += w * Fraction(num * cn_d0, comb(n, d1) * den)
    return total


def evidence_walk_product_form(
    n: int, d0: int, defects: list[int], weights: tuple[Fraction, ...] | None = None
) -> list[Fraction | float]:
    """独立重算：逐步条件概率比值的逐分量乘积（用于交叉核验，不调用 comb）。

    对每个备择 d1 维护乘积 P_{d1}(x_1..x_s)/P_{d0}(x_1..x_s)，其中单步
    条件概率 q_d(x|历史) 用无放回超几何条件：
      x=1: (d - k_prev)/slots；x=0: (slots - d + k_prev)/slots，slots = n-s+1。
    返回每个前缀 t=1..len(defects) 的 M_t。
    """
    if weights is None:
        weights = mixture_weights(d0)
    prods = [Fraction(1) for _ in weights]
    out: list[Fraction | float] = []
    k = 0
    state = "run"  # run / collapsed(证据归零) / refuted(证据无穷)
    for s, x in enumerate(defects, start=1):
        if state == "collapsed":
            out.append(Fraction(0))
            continue
        if state == "refuted":
            out.append(INF)
            continue
        k_prev = k
        k += x
        if d0 - k > n - s:
            state = "refuted"
            out.append(INF)
            continue
        if k > d0:
            state = "collapsed"
            out.append(Fraction(0))
            continue
        slots = n - s + 1
        for d1 in range(len(weights)):
            if x == 1:
                num = d1 - k_prev  # <=0 表示该备择下不可能，似然为 0
                den = d0 - k_prev  # 上述分支保证 den >= 1
                if num <= 0:
                    prods[d1] = Fraction(0)
                    continue
            else:
                num = slots - d1 + k_prev  # 上述分支保证 >= 1
                den = slots - d0 + k_prev
            prods[d1] *= Fraction(num, den)
        out.append(sum(w * p for w, p in zip(weights, prods)))
    return out


def evidence_exact_str(ev: Fraction | float) -> str:
    """证据的精确字符串：分数/整数，或 "inf"。"""
    return "inf" if ev == INF else str(ev)


def evidence_float(ev: Fraction | float) -> float | None:
    """展示用浮点值；+inf 以 null 表示（JSON 无标准无穷）。"""
    return None if ev == INF else float(ev)


def evidence_ge(ev: Fraction | float, threshold: Fraction) -> bool:
    """M_t >= 阈值 的精确判定（+inf 恒真）。"""
    return True if ev == INF else ev >= threshold
