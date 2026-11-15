"""
Dijkstra's minimum-cost path algorithm for grid-based least-cost path analysis.
Implemented from scratch using the grid_graph module for graph representation.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time

import numpy as np

from .grid_graph import (
    GridConfig,
    load_grid,
    get_neighbours,
    reconstruct_path,
    path_cost,
    path_physical_length,
)


@dataclass
class DijkstraResult:
    """Result of running Dijkstra's algorithm."""
    path: List[Tuple[int, int]]
    total_cost: float
    physical_length_m: float
    explored_nodes: int
    execution_time_s: float
    reached: bool


def dijkstra(config: GridConfig) -> DijkstraResult:
    """
    Run Dijkstra's algorithm from start to goal.
    Returns the optimal path (if any), total cost, explored node count, and timing.
    """
    start = config.start
    goal = config.goal
    H, W = config.height, config.width

    # Distance to each cell (initialized to infinity)
    dist: Dict[Tuple[int, int], float] = {}
    dist[start] = 0.0

    # Priority queue: (cost, row, col)
    pq: List[Tuple[float, int, int]] = [(0.0, start[0], start[1])]

    # To reconstruct path
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Explored node counter
    explored = 0

    # Set of visited/finalized nodes
    visited: set = set()

    start_time = time.perf_counter()

    while pq:
        cur_cost, r, c = heapq.heappop(pq)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        explored += 1

        if (r, c) == goal:
            # Found goal
            path = reconstruct_path(came_from, start, goal)
            elapsed = time.perf_counter() - start_time
            return DijkstraResult(
                path=path,
                total_cost=cur_cost,
                physical_length_m=path_physical_length(config, path),
                explored_nodes=explored,
                execution_time_s=elapsed,
                reached=True,
            )

        for nr, nc, edge_cost in get_neighbours(config, r, c):
            if (nr, nc) in visited:
                continue
            ndist = cur_cost + edge_cost
            if (nr, nc) not in dist or ndist < dist[(nr, nc)]:
                dist[(nr, nc)] = ndist
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(pq, (ndist, nr, nc))

    # Goal not reached
    elapsed = time.perf_counter() - start_time
    return DijkstraResult(
        path=[],
        total_cost=float('inf'),
        physical_length_m=0.0,
        explored_nodes=explored,
        execution_time_s=elapsed,
        reached=False,
    )


def run_on_config(config: GridConfig) -> DijkstraResult:
    """Convenience wrapper."""
    return dijkstra(config)


# --- Self-test ---
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    # Add src to path
    SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(SRC_ROOT))

    from corridor_optimizer.algorithms import load_grid, dijkstra

    tif_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "data", "raw", "synthetic",
        "synthetic_resistance_5km.tif"
    )
    try:
        cfg = load_grid(tif_path)
        print(f"Dijkstra on {cfg.height}x{cfg.width} grid, start={cfg.start}, goal={cfg.goal}")
        result = dijkstra(cfg)

        print(f"\nResult:")
        print(f"  Reached: {result.reached}")
        print(f"  Path nodes: {len(result.path)}")
        print(f"  Total resistance cost: {result.total_cost:.4f}")
        print(f"  Physical length: {result.physical_length_m:.2f} m ({result.physical_length_m/1000:.4f} km)")
        print(f"  Explored nodes: {result.explored_nodes}")
        print(f"  Execution time: {result.execution_time_s*1000:.2f} ms")

        if result.path:
            print(f"\n  First 5 path nodes: {result.path[:5]}")
            print(f"  Last 5 path nodes: {result.path[-5:]}")

        print("\n[SUCCESS] dijkstra.py verified.")
    except Exception as e:
        print(f"\n[FAIL] dijkstra.py test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)