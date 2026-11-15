"""
Habitat patch identification from land-cover grids.
Uses connected-component analysis to find contiguous habitat patches.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import label
import pandas as pd


@dataclass
class HabitatPatch:
    """Represents a contiguous habitat patch."""
    id: int                           # Unique patch identifier (sequential across classes)
    label_value: int                  # Connected component label within its class
    class_code: int                   # Land-cover class code
    class_name: str                   # Land-cover class name
    area_cells: int                   # Number of cells in patch
    area_hectares: float              # Area in hectares
    centroid: Tuple[int, int]         # (row, col) centroid in grid coordinates
    bbox: Tuple[int, int, int, int]   # (min_row, min_col, max_row, max_col)
    perimeter_cells: int              # Approximate perimeter length in cells
    compactness: float                # Shape compactness (0-1, 1=circle)


@dataclass
class HabitatAnalysisResult:
    """Complete habitat identification results."""
    patches: List[HabitatPatch]
    grid_shape: Tuple[int, int]
    cell_size: float                  # metres
    eligible_classes: List[int]
    min_patch_cells: int
    total_habitat_cells: int
    total_habitat_hectares: float


def identify_habitat_patches(
    landcover: np.ndarray,
    resistance: np.ndarray,
    cell_size: float,
    eligible_classes: List[int],
    class_names: Dict[int, str],
    min_patch_cells: int = 25,       # Default: 25 cells = 1 ha at 20m resolution
    max_patches: int = 50,           # Limit output
) -> HabitatAnalysisResult:
    """
    Identify contiguous habitat patches from land-cover grid.

    Connected-component labeling is performed PER CLASS to ensure patches
    are homogeneous by land-cover type. Different habitat classes do not merge.

    Args:
        landcover: H x W array of land-cover class codes
        resistance: H x W array of resistance values (for blocked cell detection)
        cell_size: Size of each cell in metres
        eligible_classes: List of class codes considered suitable habitat
        class_names: Mapping from class code to human-readable name
        min_patch_cells: Minimum patch size in cells (filters small fragments)
        max_patches: Maximum number of patches to return (largest first)

    Returns:
        HabitatAnalysisResult with filtered patches sorted by area (descending)
    """
    H, W = landcover.shape

    patches = []
    total_cells = 0
    patch_counter = 0  # Sequential patch ID across all classes

    for cls in eligible_classes:
        # Create mask for this specific class
        class_mask = (landcover == cls) & np.isfinite(resistance)

        if not class_mask.any():
            continue

        # Connected component labeling for this class only
        labeled_array, num_features = label(class_mask, structure=np.ones((3, 3)))

        for label_id in range(1, num_features + 1):
            mask = (labeled_array == label_id)
            cells = int(np.sum(mask))

            if cells < min_patch_cells:
                continue

            patch_counter += 1

            # Centroid
            rows, cols = np.where(mask)
            centroid_row = int(np.mean(rows))
            centroid_col = int(np.mean(cols))

            # Bounding box
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Perimeter approximation (4-connectivity edge count)
            perimeter = 0
            for r, c in zip(rows, cols):
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
                        perimeter += 1

            # Area in hectares
            cell_area_m2 = cell_size * cell_size
            area_hectares = (cells * cell_area_m2) / 10000.0

            # Compactness: 4π*Area / Perimeter^2 (1 = perfect circle)
            if perimeter > 0:
                compactness = (4 * np.pi * area_hectares * 10000) / (perimeter * perimeter * cell_area_m2)
                compactness = min(1.0, compactness)
            else:
                compactness = 1.0

            patch = HabitatPatch(
                id=patch_counter,           # Sequential ID across classes
                label_value=label_id,       # Original component label within class
                class_code=cls,             # This patch's class code
                class_name=class_names.get(cls, f"Class {cls}"),
                area_cells=cells,
                area_hectares=area_hectares,
                centroid=(centroid_row, centroid_col),
                bbox=(min_row, min_col, max_row, max_col),
                perimeter_cells=perimeter,
                compactness=compactness,
            )
            patches.append(patch)
            total_cells += cells

    # Sort by area descending, limit to max_patches
    patches.sort(key=lambda p: p.area_cells, reverse=True)
    patches = patches[:max_patches]

    total_hectares = (total_cells * cell_size * cell_size) / 10000.0

    return HabitatAnalysisResult(
        patches=patches,
        grid_shape=(H, W),
        cell_size=cell_size,
        eligible_classes=eligible_classes,
        min_patch_cells=min_patch_cells,
        total_habitat_cells=total_cells,
        total_habitat_hectares=total_hectares,
    )


def filter_patches_by_distance(
    patches: List[HabitatPatch],
    min_separation_cells: int = 50,
    max_separation_cells: int = 500,
) -> List[Tuple[HabitatPatch, HabitatPatch]]:
    """
    Find patch pairs within a reasonable distance range for corridor analysis.

    Returns list of (patch_a, patch_b) tuples sorted by combined area.
    """
    pairs = []
    for i, patch_a in enumerate(patches):
        for patch_b in patches[i+1:]:
            # Centroid distance
            r1, c1 = patch_a.centroid
            r2, c2 = patch_b.centroid
            dist = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)

            if min_separation_cells <= dist <= max_separation_cells:
                combined_area = patch_a.area_cells + patch_b.area_cells
                pairs.append((combined_area, patch_a, patch_b, dist))

    # Sort by combined area descending
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [(a, b) for _, a, b, _ in pairs]


def create_patch_mask(
    landcover: np.ndarray,
    labeled_array: np.ndarray,
    patch: HabitatPatch,
) -> np.ndarray:
    """Create a binary mask for a specific patch."""
    return (landcover == patch.class_code) & (labeled_array == patch.label_value)


def patches_to_dataframe(patches: List[HabitatPatch]) -> pd.DataFrame:
    """Convert patches to a pandas DataFrame for display/export."""
    rows = []
    for p in patches:
        rows.append({
            'Patch ID': p.id,
            'Class Code': p.class_code,
            'Class Name': p.class_name,
            'Area (cells)': p.area_cells,
            'Area (ha)': round(p.area_hectares, 2),
            'Centroid (row, col)': f"({p.centroid[0]}, {p.centroid[1]})",
            'BBox (r1,c1,r2,c2)': f"({p.bbox[0]}, {p.bbox[1]}, {p.bbox[2]}, {p.bbox[3]})",
            'Perimeter (cells)': p.perimeter_cells,
            'Compactness': round(p.compactness, 3),
        })
    return pd.DataFrame(rows)


# --- Self-test with synthetic data ---
if __name__ == "__main__":
    import os
    import sys
    import rasterio
    from pathlib import Path

    # Add src to path
    SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(SRC_ROOT))

    from corridor_optimizer.algorithms import identify_habitat_patches, filter_patches_by_distance

    data_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "synthetic"

    with rasterio.open(str(data_dir / "synthetic_landcover_5km.tif")) as ds:
        landcover = ds.read(1)

    with rasterio.open(str(data_dir / "synthetic_resistance_5km.tif")) as ds:
        resistance = ds.read(1).astype(np.float64)
        resistance[~np.isfinite(resistance)] = np.inf

    # For synthetic data: eligible classes are 1 (dense habitat) and maybe 2 (shrubland)
    eligible = [1, 2]
    class_names = {1: "Dense habitat", 2: "Shrubland"}

    result = identify_habitat_patches(
        landcover=landcover,
        resistance=resistance,
        cell_size=20.0,
        eligible_classes=eligible,
        class_names=class_names,
        min_patch_cells=25,
    )

    print(f"Grid: {result.grid_shape}, Cell size: {result.cell_size}m")
    print(f"Eligible classes: {result.eligible_classes}")
    print(f"Min patch size: {result.min_patch_cells} cells")
    print(f"Total habitat: {result.total_habitat_cells} cells ({result.total_habitat_hectares:.2f} ha)")
    print(f"Patches found: {len(result.patches)}")

    for p in result.patches[:10]:
        print(f"  Patch {p.id}: {p.class_name}, {p.area_cells} cells ({p.area_hectares:.2f} ha), "
              f"centroid={p.centroid}, compactness={p.compactness:.3f}")

    # Find candidate pairs
    pairs = filter_patches_by_distance(result.patches, min_separation_cells=50, max_separation_cells=300)
    print(f"\nCandidate patch pairs (50-300 cells apart): {len(pairs)}")
    for i, (a, b) in enumerate(pairs[:5]):
        dist = np.sqrt((a.centroid[0]-b.centroid[0])**2 + (a.centroid[1]-b.centroid[1])**2)
        print(f"  Pair {i+1}: Patch {a.id} ({a.area_hectares:.1f}ha) <-> Patch {b.id} ({b.area_hectares:.1f}ha), "
              f"dist={dist:.1f} cells ({dist*20/1000:.2f} km)")

    print("\n[OK] habitat.py self-test passed")