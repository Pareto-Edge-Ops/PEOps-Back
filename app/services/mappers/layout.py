"""Topology → 3D layout (depth / col / zCol) for the architecture viz.

depth  = longest path from any source over the kept-node DAG
col    = symmetric signed y-offset for parallel branches at the same depth
zCol   = second grid axis when a level holds > GRID_THRESHOLD siblings
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COL_GAP = 3.4
GRID_THRESHOLD = 4


@dataclass
class NodePos:
    depth: int
    col: float
    z_col: float | None


def compute_layout(
    node_ids: list[str],
    edges: list[tuple[str, str]],
    order_index: dict[str, int],
) -> dict[str, NodePos]:
    """`node_ids` in topological order; `edges` over kept nodes only.
    `order_index` provides a stable sibling sort key (topo index)."""
    preds: dict[str, list[str]] = {n: [] for n in node_ids}
    for frm, to in edges:
        if frm in preds and to in preds:
            preds[to].append(frm)

    depth: dict[str, int] = {}
    for n in node_ids:  # already topologically sorted
        ps = preds[n]
        depth[n] = 0 if not ps else 1 + max(depth[p] for p in ps if p in depth)

    levels: dict[int, list[str]] = {}
    for n in node_ids:
        levels.setdefault(depth[n], []).append(n)

    pos: dict[str, NodePos] = {}
    for d, members in levels.items():
        members.sort(key=lambda n: order_index.get(n, 0))
        k = len(members)
        if k == 1:
            pos[members[0]] = NodePos(depth=d, col=0.0, z_col=None)
        elif k <= GRID_THRESHOLD:
            for i, n in enumerate(members):
                col = round((i - (k - 1) / 2) * COL_GAP, 3)
                pos[n] = NodePos(depth=d, col=col, z_col=None)
        else:
            cols = math.ceil(math.sqrt(k))
            rows = math.ceil(k / cols)
            for i, n in enumerate(members):
                r, c = divmod(i, cols)
                pos[n] = NodePos(
                    depth=d,
                    col=round((c - (cols - 1) / 2) * COL_GAP, 3),
                    z_col=round((r - (rows - 1) / 2) * COL_GAP, 3),
                )
    return pos
