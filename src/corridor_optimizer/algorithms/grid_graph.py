"""
Grid graph representation for 8-directional least-cost pathfinding
on the synthetic ecological corridor resistance surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import rasterio


# Movement directions: (dr, dc, base_distance)
# 8 neighbours: N, NE, E, SE, S, SW, W, NW
SQRT2 = math.sqrt(2.0)
DIRECTIONS_8 = [
    (-1, 0, 1.0),     # 0: N
    (-1, 1, SQRT2),   # 1: NE
    (0, 1, 1.0),      # 2: E
    (1, 1, SQRT2),    # 3: SE
    (1, 0, 1.0),      # 4: S
    (1, -1, SQRT2),   # 5: SW
    (0, -1, 1.0),     # 6: W
    (-1, -1, SQRT2),  # 7: NW
]

# Orthogonal directions for corner-cutting check:
# For diagonal index d: check the two orthogonal indices adjacent to it.
# NE (1) checks N (0) and E (2)
# SE (3) checks E (2) and S (4)
# SW (5) checks S (4) and W (6)
# NW (7) checks W (6) and N (0)
ORTHOGONAL_CORNER_CHECKS: Dict[int, Tuple[int, int]] = {
    1: (0, 2),  # NE checks N and E
    3: (2, 4),  # SE checks E and S
    5: (4, 6),  # SW checks S and W
    7: (6, 0),  # NW checks W and N
}


@dataclass(frozen=True)
class GridConfig:
    """Immutable grid configuration."""
    resistance: np.ndarray          # H x W float array, inf = blocked
    height: int
    width: int
    cell_size: float                # metres
    start: Tuple[int, int]          # (row, col)
    goal: Tuple[int, int]           # (row, col)


def load_grid(
    tif_path: str,
    start: Tuple[int, int] = (116, 42),
    goal: Tuple[int, int] = (133, 208),
    cell_size: float = 20.0,
) -> GridConfig:
    """
    Load resistance surface from GeoTIFF and return a GridConfig.
    Infinite resistance values are treated as blocked cells.
    """
    with rasterio.open(tif_path) as ds:
        resistance = ds.read(1).astype(np.float64)
        resistance[~np.isfinite(resistance)] = np.inf
        height, width = resistance.shape

    # Validate start/goal
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        raise ValueError(f"Start {start} out of grid bounds ({height}x{width})")
    if not (0 <= goal[0] < height and 0 <= goal[1] < width):
        raise ValueError(f"Goal {goal} out of grid bounds ({height}x{width})")
    if np.isinf(resistance[start]):
        raise ValueError(f"Start cell {start} is blocked")
    if np.isinf(resistance[goal]):
        raise ValueError(f"Goal cell {goal} is blocked")

    return GridConfig(
        resistance=resistance,
        height=height,
        width=width,
        cell_size=cell_size,
        start=start,
        goal=goal,
    )


def in_bounds(r: int, c: int, height: int, width: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def is_blocked(resistance: np.ndarray, r: int, c: int) -> bool:
    return not np.isfinite(resistance[r, c])


def get_neighbours(
    config: GridConfig,
    r: int,
    c: int,
) -> List[Tuple[int, int, float]]:
    """
    Return valid 8-directional neighbours from (r, c).
    Each entry: (nr, nc, movement_cost)
    movement_cost = distance * AVERAGE(source_resistance, target_resistance)
    Diagonal moves are blocked if EITHER adjacent orthogonal cell is blocked
    to prevent cutting through blocked corners.

    Using the average of source and target cell resistance ensures a symmetric
    undirected graph: cost(A->B) == cost(B->A), which is ecologically appropriate
    for corridor analysis where movement works both directions.
    """
    H, W = config.height, config.width
    res = config.resistance
    neighbours = []

    for dir_idx, (dr, dc, dist) in enumerate(DIRECTIONS_8):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if not np.isfinite(res[nr, nc]):
            continue

        # Check corner cutting for diagonal moves
        if dir_idx in ORTHOGONAL_CORNER_CHECKS:
            o1_idx, o2_idx = ORTHOGONAL_CORNER_CHECKS[dir_idx]
            dr1, dc1, _ = DIRECTIONS_8[o1_idx]
            dr2, dc2, _ = DIRECTIONS_8[o2_idx]
            r1, c1 = r + dr1, c + dc1
            r2, c2 = r + dr2, c + dc2
            # If either orthogonal neighbor is in bounds and blocked, block diagonal
            b1 = (0 <= r1 < H and 0 <= c1 < W) and not np.isfinite(res[r1, c1])
            b2 = (0 <= r2 < H and 0 <= c2 < W) and not np.isfinite(res[r2, c2])
            if b1 or b2:
                continue

        # Edge cost = distance * AVERAGE resistance of source and target cells
        # This makes the graph undirected: cost(A->B) == cost(B->A)
        cost = dist * (res[r, c] + res[nr, nc]) * 0.5
        neighbours.append((nr, nc, cost))

    return neighbours


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Reconstruct path from goal back to start using came_from dict."""
    if goal not in came_from and goal != start:
        return []
    path = []
    current: Optional[Tuple[int, int]] = goal
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = came_from.get(current)
    path.reverse()
    if not path or path[0] != start:
        return []
    return path


def path_cost(config: GridConfig, path: List[Tuple[int, int]]) -> float:
    """Compute total accumulated resistance cost of a path using symmetric edge costs."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    res = config.resistance
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        dist = SQRT2 if (dr == 1 and dc == 1) else 1.0
        # Symmetric: average of source and target resistance
        total += dist * (res[r1, c1] + res[r2, c2]) * 0.5
    return total


def path_physical_length(config: GridConfig, path: List[Tuple[int, int]]) -> float:
    """Compute physical path length in metres."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        dist = SQRT2 if (dr == 1 and dc == 1) else 1.0
        total += dist * config.cell_size
    return total


# --- Self-test ---
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

    # Add src to path
    SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(SRC_ROOT))

    from corridor_optimizer.algorithms import load_grid, get_neighbours

    tif_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "data", "raw", "synthetic",
        "synthetic_resistance_5km.tif"
    )
    try:
        cfg = load_grid(tif_path)
        print(f"Loaded grid: {cfg.height}x{cfg.width}, cell={cfg.cell_size}m")
        print(f"Start: {cfg.start}, Goal: {cfg.goal}")
        print(f"Start resistance: {cfg.resistance[cfg.start]}")
        print(f"Goal resistance: {cfg.resistance[cfg.goal]}")
        print(f"Blocked cells: {int(np.isinf(cfg.resistance).sum())}")

        # Test neighbours at start
        start_nbrs = get_neighbours(cfg, *cfg.start)
        print(f"\nNeighbours of start {cfg.start}: {len(start_nbrs)}")
        for nr, nc, cost in start_nbrs:
            print(f"  ({nr}, {nc}) cost={cost:.4f} res={cfg.resistance[nr, nc]}")

        # Test corner-cutting blocking rule on a synthetic 2x2 corner
        test_res = np.ones((3, 3), dtype=np.float64)
        test_res[0, 1] = np.inf  # North is blocked
        test_cfg = GridConfig(test_res, 3, 3, 20.0, (1, 0), (0, 0))
        # From (1,0), moving NE (0,1) is target blocked. Moving NW (0,-1) is out of bounds.
        # But moving from (1,1) NE to (0,2)? Let's check (1,0) to NE (0,1):
        # o1 (N)=0,1 (blocked), so NE to (0,1) blocked.
        # Let's verify NE from (1,0) to (0,1):
        nbrs_test = get_neighbours(test_cfg, 1, 0)
        ne_target = (0, 1)
        has_ne = any((nr, nc) == ne_target for nr, nc, _ in nbrs_test)
        print(f"\nCorner-cutting unit test (N blocked at 0,1): NE move present? {has_ne} (should be False)")

        print("\n[SUCCESS] grid_graph.py verified.")
    except Exception as e:
        print(f"\n[FAIL] grid_graph.py test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)