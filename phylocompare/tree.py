"""无根二叉树的内部表示与校验。

候选树在 Newick 中以三叉根序列化（根没有枝长、不是真实节点），
展开后是一棵无根二叉树：n 个叶、n-2 个度为 3 的内部节点、2n-3 条边。
边以其诱导的叶集划分做规范编号，使 Newick 书写顺序、叶排列变化
都不会改变参数与结果的含义。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CompareError
from .newick import Node


@dataclass(frozen=True)
class Edge:
    """一条无向边；key 是较小一侧的排序叶元组，作为规范身份。"""

    index: int
    key: tuple[str, ...]
    endpoints: tuple[int, int]
    initial_length: float


class UnrootedTree:
    def __init__(
        self,
        leaves: tuple[str, ...],
        edges: list[Edge],
        adjacency: dict[int, list[tuple[int, int]]],
        internal_count: int,
    ) -> None:
        self.leaves = leaves
        self.leaf_index = {name: i for i, name in enumerate(leaves)}
        self.edges = edges
        self.adjacency = adjacency
        self.internal_count = internal_count
        self.topology_key = tuple(e.key for e in edges)
        self.initial_lengths = tuple(e.initial_length for e in edges)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # ---- 构造 ----------------------------------------------------------

    @classmethod
    def from_newick(cls, root: Node, expected_leaves: tuple[str, ...]) -> "UnrootedTree":
        """把三叉根 Newick 树展开为无根二叉树，并完成结构校验。"""
        if root.is_leaf:
            raise CompareError("候选树根节点必须是内部节点")
        if len(root.children) != 3:
            raise CompareError(
                f"无根二叉树应以三叉根序列化，根的子节点数为 {len(root.children)} 而不是 3"
            )
        if root.length is not None:
            raise CompareError("三叉根不应带有枝长")

        # 收集叶名并检查唯一性。
        names: list[str] = []

        def walk(node: Node, depth: int = 1) -> None:
            if node.is_leaf:
                if not node.name:
                    raise CompareError("存在没有名称的叶节点")
                names.append(node.name)
                return
            if depth > 1 and len(node.children) != 2:
                raise CompareError(
                    f"内部节点 {node.name or '<unnamed>'} 应为二叉（2 个子节点），"
                    f"实际有 {len(node.children)} 个"
                )
            for child in node.children:
                if child.length is None:
                    raise CompareError("每条边都必须给出初始枝长")
                if child.length <= 0:
                    raise CompareError(
                        f"边 {node.name or '<root>'}->{child.name or '<node>'} 的初始枝长"
                        f"必须为正数，得到 {child.length}"
                    )
                walk(child, depth + 1)

        walk(root)
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise CompareError(f"树内叶名不唯一，重复叶: {', '.join(dup)}")
        leaf_set = set(names)
        expected_set = set(expected_leaves)
        if leaf_set != expected_set:
            missing = sorted(expected_set - leaf_set)
            extra = sorted(leaf_set - expected_set)
            detail = []
            if missing:
                detail.append(f"缺少 {missing}")
            if extra:
                detail.append(f"多出 {extra}")
            raise CompareError("候选树叶集合与比对不一致: " + "; ".join(detail))

        leaves = tuple(sorted(names))

        # 给所有节点分配整数 id：叶用 0..n-1，内部节点随后编号。
        degree: dict[int, int] = {}
        raw_edges: list[tuple[int, int, float]] = []
        next_internal = [len(leaves)]

        def assign(node: Node) -> int:
            if node.is_leaf:
                return leaves.index(node.name)  # type: ignore[arg-type]
            nid = next_internal[0]
            next_internal[0] += 1
            for child in node.children:
                cid = assign(child)
                raw_edges.append((nid, cid, float(child.length)))  # type: ignore[arg-type]
            return nid

        # 根本身是表示无根树的“虚拟三叉点”，需要作为一个真实内部节点。
        root_id = next_internal[0]
        next_internal[0] += 1
        for child in root.children:
            cid = assign(child)
            raw_edges.append((root_id, cid, float(child.length)))  # type: ignore[arg-type]

        adjacency: dict[int, list[tuple[int, int]]] = {}
        for u, v, length in raw_edges:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            adjacency.setdefault(u, []).append((v, length))
            adjacency.setdefault(v, []).append((u, length))

        internal_count = next_internal[0] - len(leaves)
        for nid, deg in degree.items():
            if nid < len(leaves):
                if deg != 1:
                    raise CompareError(f"叶 {leaves[nid]} 的度应为 1，实际为 {deg}")
            elif deg != 3:
                raise CompareError(f"内部节点的度应为 3，实际为 {deg}")
        expected_edges = 2 * len(leaves) - 3
        if len(raw_edges) != expected_edges:
            raise CompareError(
                f"{len(leaves)} 叶无根二叉树应有 {expected_edges} 条边，"
                f"实际解析出 {len(raw_edges)} 条"
            )

        # 用边两侧的叶集合生成规范 key，再按 key 稳定编号。
        def side_leaves(start: int, via: int) -> tuple[str, ...]:
            seen: list[str] = []
            stack = [(start, via)]
            while stack:
                node, parent = stack.pop()
                if node < len(leaves):
                    seen.append(leaves[node])
                    continue
                for nbr, _ in adjacency[node]:
                    if nbr != parent:
                        stack.append((nbr, node))
            return tuple(sorted(seen))

        canonical: list[tuple[tuple[str, ...], int, int, float]] = []
        for u, v, length in raw_edges:
            side_u = side_leaves(u, v)
            side_v = side_leaves(v, u)
            key = side_u if len(side_u) <= len(side_v) else side_v
            canonical.append((key, u, v, length))
        canonical.sort(key=lambda item: item[0])

        edges = [
            Edge(index=i, key=key, endpoints=(u, v), initial_length=length)
            for i, (key, u, v, length) in enumerate(canonical)
        ]
        edge_index_by_pair = {
            frozenset((u, v)): edge.index for edge in edges for u, v in (edge.endpoints,)
        }
        adjacency_by_edge: dict[int, list[tuple[int, int]]] = {
            nid: [(nbr, edge_index_by_pair[frozenset((nid, nbr))]) for nbr, _ in nbrs]
            for nid, nbrs in adjacency.items()
        }
        return cls(leaves, edges, adjacency_by_edge, internal_count)

    # ---- 供模型使用的工具 ----------------------------------------------

    def clipped_initial_lengths(self, bounds: tuple[float, float]) -> tuple[float, ...]:
        """公开规则：初始枝长若落在声明界外，先裁到界内再参与多起点优化。"""
        lo, hi = bounds
        return tuple(min(hi, max(lo, length)) for length in self.initial_lengths)

    def reorder_leaf_sequences(self, order: tuple[str, ...]) -> "UnrootedTree":
        """按给定叶顺序重建映射（用于证明叶排列不变性，拓扑本身不变）。"""
        if set(order) != set(self.leaves):
            raise CompareError("叶排列变化不能改变叶集合")
        remap = {self.leaf_index[name]: i for i, name in enumerate(order)}
        remap.update(
            {
                len(self.leaves) + k: len(order) + k
                for k in range(self.internal_count)
            }
        )
        new_edges = [
            Edge(
                index=edge.index,
                key=edge.key,
                endpoints=(remap[edge.endpoints[0]], remap[edge.endpoints[1]]),
                initial_length=edge.initial_length,
            )
            for edge in self.edges
        ]
        new_adj: dict[int, list[tuple[int, int]]] = {}
        for nid, nbrs in self.adjacency.items():
            new_adj.setdefault(remap[nid], []).extend(
                (remap[nbr], edge_idx) for nbr, edge_idx in nbrs
            )
        return UnrootedTree(order, new_edges, new_adj, self.internal_count)
