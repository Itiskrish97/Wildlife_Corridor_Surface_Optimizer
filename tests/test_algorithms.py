"""Tests for corridor_optimizer algorithms."""

import numpy as np
import pytest
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
import sys
sys.path.insert(0, str(SRC_ROOT))

from corridor_optimizer.algorithms import (
    GridConfig,
    get_neighbours,
    reconstruct_path,
    path_cost,
    path_physical_length,
    dijkstra,
    DijkstraResult,
    identify_habitat_patches,
    filter_patches_by_distance,
    HabitatPatch,
    HabitatAnalysisResult,
    SQRT2,
    DIRECTIONS_8,
    ORTHOGONAL_CORNER_CHECKS,
)


class TestGridGraph:
    """Tests for grid_graph module."""

    def test_sqrt2(self):
        assert SQRT2 == pytest.approx(1.4142135623730951)

    def test_directions_8_length(self):
        assert len(DIRECTIONS_8) == 8

    def test_directions_8_values(self):
        # N
        assert DIRECTIONS_8[0] == (-1, 0, 1.0)
        # NE
        assert DIRECTIONS_8[1] == (-1, 1, SQRT2)
        # E
        assert DIRECTIONS_8[2] == (0, 1, 1.0)
        # SE
        assert DIRECTIONS_8[3] == (1, 1, SQRT2)
        # S
        assert DIRECTIONS_8[4] == (1, 0, 1.0)
        # SW
        assert DIRECTIONS_8[5] == (1, -1, SQRT2)
        # W
        assert DIRECTIONS_8[6] == (0, -1, 1.0)
        # NW
        assert DIRECTIONS_8[7] == (-1, -1, SQRT2)

    def test_corner_checks(self):
        assert ORTHOGONAL_CORNER_CHECKS == {
            1: (0, 2),  # NE checks N and E
            3: (2, 4),  # SE checks E and S
            5: (4, 6),  # SW checks S and W
            7: (6, 0),  # NW checks W and N
        }

    def test_get_neighbours_basic(self):
        """Test basic neighbour retrieval on a simple grid."""
        # 3x3 grid, all resistance = 1.0, cell_size = 20
        resistance = np.ones((3, 3), dtype=np.float64)
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(1, 1),
            goal=(2, 2),
        )

        # Center cell (1,1) should have 8 neighbours
        neighbours = get_neighbours(config, 1, 1)
        assert len(neighbours) == 8

        # All costs should be distance * (1+1)/2 = distance
        for nr, nc, cost in neighbours:
            dr = abs(nr - 1)
            dc = abs(nc - 1)
            expected_dist = SQRT2 if (dr == 1 and dc == 1) else 1.0
            assert cost == pytest.approx(expected_dist)

    def test_get_neighbours_corner_cutting(self):
        """Test that diagonal moves are blocked when orthogonal neighbours are blocked."""
        # 3x3 grid with blocked north cell
        resistance = np.ones((3, 3), dtype=np.float64)
        resistance[0, 1] = np.inf  # North is blocked
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(1, 1),
            goal=(2, 2),
        )

        # From center (1,1), NE should be blocked (north is blocked)
        neighbours = get_neighbours(config, 1, 1)
        ne_exists = any(nr == 0 and nc == 2 for nr, nc, _ in neighbours)
        assert not ne_exists, "NE should be blocked when N is blocked"

        # But NW should still work (west is not blocked)
        nw_exists = any(nr == 0 and nc == 0 for nr, nc, _ in neighbours)
        assert nw_exists, "NW should work when W is not blocked"

    def test_get_neighbours_edge_cells(self):
        """Test neighbours at grid edges."""
        resistance = np.ones((3, 3), dtype=np.float64)
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(0, 0),
            goal=(2, 2),
        )

        # Corner (0,0) should have 3 neighbours: E, SE, S
        neighbours = get_neighbours(config, 0, 0)
        assert len(neighbours) == 3
        coords = {(nr, nc) for nr, nc, _ in neighbours}
        assert coords == {(0, 1), (1, 1), (1, 0)}

    def test_reconstruct_path(self):
        """Test path reconstruction."""
        came_from = {
            (1, 1): (0, 0),
            (2, 2): (1, 1),
            (3, 3): (2, 2),
        }
        start = (0, 0)
        goal = (3, 3)

        path = reconstruct_path(came_from, start, goal)
        assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_reconstruct_path_empty(self):
        """Test path reconstruction when goal not reached."""
        came_from = {}
        start = (0, 0)
        goal = (1, 1)

        path = reconstruct_path(came_from, start, goal)
        assert path == []

    def test_path_cost(self):
        """Test path cost calculation."""
        resistance = np.ones((3, 3), dtype=np.float64)
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(0, 0),
            goal=(2, 2),
        )

        # Straight line: (0,0) -> (1,0) -> (2,0) = 2 steps * 1.0 * 1.0 = 2.0
        path = [(0, 0), (1, 0), (2, 0)]
        cost = path_cost(config, path)
        assert cost == pytest.approx(2.0)

        # Diagonal: (0,0) -> (1,1) = 1 step * sqrt(2) * 1.0 = sqrt(2)
        path = [(0, 0), (1, 1)]
        cost = path_cost(config, path)
        assert cost == pytest.approx(SQRT2)

    def test_path_physical_length(self):
        """Test physical path length calculation."""
        resistance = np.ones((3, 3), dtype=np.float64)
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(0, 0),
            goal=(2, 2),
        )

        # Straight line: 2 * 20 = 40m
        path = [(0, 0), (1, 0), (2, 0)]
        length = path_physical_length(config, path)
        assert length == pytest.approx(40.0)

        # Diagonal: sqrt(2) * 20
        path = [(0, 0), (1, 1)]
        length = path_physical_length(config, path)
        assert length == pytest.approx(20.0 * SQRT2)


class TestDijkstra:
    """Tests for dijkstra module."""

    def test_simple_path(self):
        """Test Dijkstra on a simple 3x3 grid."""
        resistance = np.ones((3, 3), dtype=np.float64)
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(0, 0),
            goal=(2, 2),
        )

        result = dijkstra(config)
        assert result.reached
        assert len(result.path) == 3  # (0,0) -> (1,1) -> (2,2)
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (2, 2)
        assert result.total_cost == pytest.approx(2 * SQRT2)

    def test_blocked_goal(self):
        """Test Dijkstra when goal is blocked."""
        resistance = np.ones((3, 3), dtype=np.float64)
        resistance[2, 2] = np.inf
        config = GridConfig(
            resistance=resistance,
            height=3,
            width=3,
            cell_size=20.0,
            start=(0, 0),
            goal=(2, 2),
        )

        result = dijkstra(config)
        assert not result.reached
        assert result.path == []
        assert result.total_cost == float('inf')

    def test_obstacle_avoidance(self):
        """Test that Dijkstra routes around obstacles."""
        # 5x5 grid with vertical barrier in middle
        resistance = np.ones((5, 5), dtype=np.float64)
        resistance[:, 2] = np.inf  # Block column 2
        resistance[2, 2] = 1.0     # But leave a gap at row 2

        config = GridConfig(
            resistance=resistance,
            height=5,
            width=5,
            cell_size=20.0,
            start=(2, 0),
            goal=(2, 4),
        )

        result = dijkstra(config)
        assert result.reached
        # Path should go through the gap at (2,2)
        assert (2, 2) in result.path


class TestHabitat:
    """Tests for habitat module."""

    def test_identify_habitat_patches_simple(self):
        """Test habitat patch identification on simple grid."""
        # 5x5 grid: class 1 in top-left 3x3, class 2 in bottom-right 2x2
        landcover = np.zeros((5, 5), dtype=int)
        landcover[:3, :3] = 1  # Class 1
        landcover[3:, 3:] = 2  # Class 2

        resistance = np.ones((5, 5), dtype=np.float64)

        result = identify_habitat_patches(
            landcover=landcover,
            resistance=resistance,
            cell_size=20.0,
            eligible_classes=[1, 2],
            class_names={1: "Class 1", 2: "Class 2"},
            min_patch_cells=1,
        )

        assert len(result.patches) == 2
        assert result.total_habitat_cells == 13  # 9 + 4
        assert result.total_habitat_hectares == pytest.approx(13 * 400 / 10000)

        # Check patch 1 (larger)
        p1 = result.patches[0]
        assert p1.class_code == 1
        assert p1.area_cells == 9

        # Check patch 2
        p2 = result.patches[1]
        assert p2.class_code == 2
        assert p2.area_cells == 4

    def test_identify_habitat_patches_min_size_filter(self):
        """Test that small patches are filtered out."""
        landcover = np.zeros((5, 5), dtype=int)
        landcover[0, 0] = 1  # Single cell patch
        landcover[1:4, 1:4] = 1  # 3x3 patch

        resistance = np.ones((5, 5), dtype=np.float64)

        result = identify_habitat_patches(
            landcover=landcover,
            resistance=resistance,
            cell_size=20.0,
            eligible_classes=[1],
            class_names={1: "Class 1"},
            min_patch_cells=5,  # Filter out single cell
        )

        assert len(result.patches) == 1
        assert result.patches[0].area_cells == 9

    def test_identify_habitat_patches_per_class(self):
        """Test that different classes don't merge."""
        # Two adjacent patches of different classes
        landcover = np.zeros((3, 6), dtype=int)
        landcover[:, :3] = 1  # Left half class 1
        landcover[:, 3:] = 2  # Right half class 2

        resistance = np.ones((3, 6), dtype=np.float64)

        result = identify_habitat_patches(
            landcover=landcover,
            resistance=resistance,
            cell_size=20.0,
            eligible_classes=[1, 2],
            class_names={1: "Class 1", 2: "Class 2"},
            min_patch_cells=1,
        )

        # Should be 2 patches, not 1 merged patch
        assert len(result.patches) == 2
        class_codes = {p.class_code for p in result.patches}
        assert class_codes == {1, 2}

    def test_filter_patches_by_distance(self):
        """Test patch pair filtering by distance."""
        patches = [
            HabitatPatch(id=1, label_value=1, class_code=1, class_name="A",
                         area_cells=100, area_hectares=4.0, centroid=(10, 10),
                         bbox=(0, 0, 20, 20), perimeter_cells=80, compactness=0.5),
            HabitatPatch(id=2, label_value=1, class_code=1, class_name="A",
                         area_cells=50, area_hectares=2.0, centroid=(100, 100),
                         bbox=(90, 90, 110, 110), perimeter_cells=80, compactness=0.5),
            HabitatPatch(id=3, label_value=1, class_code=2, class_name="B",
                         area_cells=200, area_hectares=8.0, centroid=(20, 20),
                         bbox=(10, 10, 30, 30), perimeter_cells=80, compactness=0.5),
        ]

        # min=15, max=50: patches 1&3 are ~14 apart (in range), 1&2 are ~127 apart (out), 2&3 are ~113 apart (out)
        pairs = filter_patches_by_distance(patches, min_separation_cells=15, max_separation_cells=50)
        assert len(pairs) == 0  # 1&3 is ~14.14, just under 15

        pairs = filter_patches_by_distance(patches, min_separation_cells=10, max_separation_cells=50)
        assert len(pairs) == 1
        assert pairs[0] == (patches[0], patches[2])  # 1 and 3, sorted by combined area


if __name__ == "__main__":
    pytest.main([__file__, "-v"])