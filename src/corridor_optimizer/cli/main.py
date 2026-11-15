#!/usr/bin/env python3
"""
Ecological Corridor Optimizer - Main Entry Point

Least-cost ecological corridor identification using Dijkstra's algorithm
on ESA WorldCover 2021 resistance surfaces.

This script loads pre-computed real data outputs and runs verification.
For full pipeline execution, see the algorithms/ module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# Add src to path for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from corridor_optimizer.algorithms import (
    GridConfig,
    load_grid,
    dijkstra,
    DijkstraResult,
    path_cost,
    path_physical_length,
    identify_habitat_patches,
    filter_patches_by_distance,
    patches_to_dataframe,
)


def get_data_paths() -> dict:
    """Get standardized data paths relative to project root."""
    return {
        "raw_real": PROJECT_ROOT / "data" / "raw" / "real",
        "processed_real": PROJECT_ROOT / "data" / "processed" / "real",
        "raw_synthetic": PROJECT_ROOT / "data" / "raw" / "synthetic",
        "processed_synthetic": PROJECT_ROOT / "data" / "processed" / "synthetic",
        "outputs_figures": PROJECT_ROOT / "outputs" / "figures",
    }


def run_synthetic_regression() -> bool:
    """Run the synthetic dataset regression test."""
    print("=" * 60)
    print("SYNTHETIC DATASET REGRESSION TEST")
    print("=" * 60)

    paths = get_data_paths()
    tif_path = paths["raw_synthetic"] / "synthetic_resistance_5km.tif"

    if not tif_path.exists():
        print(f"Synthetic data not found at {tif_path}")
        return False

    cfg = load_grid(str(tif_path))
    print(f"Grid: {cfg.height}x{cfg.width}, cell={cfg.cell_size}m")
    print(f"Start: {cfg.start}, Goal: {cfg.goal}")

    result = dijkstra(cfg)

    print(f"\nResult:")
    print(f"  Reached: {result.reached}")
    print(f"  Path cells: {len(result.path)}")
    print(f"  Resistance cost: {result.total_cost:.4f}")
    print(f"  Physical length: {result.physical_length_m:.2f} m ({result.physical_length_m/1000:.3f} km)")
    print(f"  Explored nodes: {result.explored_nodes}")
    print(f"  Execution time: {result.execution_time_s*1000:.1f} ms")

    # Verify against baseline (updated for symmetric edge costs)
    expected_cost = 421.4874
    expected_len = 4381.0
    expected_cells = 172

    print(f"\nBaseline verification:")
    cost_ok = abs(result.total_cost - expected_cost) < 0.01
    len_ok = abs(result.physical_length_m - expected_len) < 1
    cells_ok = len(result.path) == expected_cells
    print(f"  Cost: {result.total_cost:.4f} (expected ~{expected_cost:.4f}) {'OK' if cost_ok else 'FAIL'}")
    print(f"  Length: {result.physical_length_m:.2f} (expected ~{expected_len:.2f}) {'OK' if len_ok else 'FAIL'}")
    print(f"  Cells: {len(result.path)} (expected {expected_cells}) {'OK' if cells_ok else 'FAIL'}")

    all_ok = cost_ok and len_ok and cells_ok
    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


def run_real_data_verification() -> bool:
    """Verify the real data pipeline outputs."""
    print("=" * 60)
    print("REAL DATA PIPELINE VERIFICATION")
    print("=" * 60)

    paths = get_data_paths()
    processed_real = paths["processed_real"]

    # Load resistance surface
    resistance_tif = processed_real / "real_resistance.tif"
    landcover_tif = processed_real / "real_landcover.tif"
    path_csv = processed_real / "dijkstra_path.csv"
    patches_csv = processed_real / "candidate_habitat_patches.csv"
    metadata_json = processed_real / "real_grid_metadata.json"

    for f in [resistance_tif, landcover_tif, path_csv, patches_csv, metadata_json]:
        if not f.exists():
            print(f"Missing file: {f}")
            return False

    # Load resistance grid
    with rasterio.open(str(resistance_tif)) as ds:
        resistance = ds.read(1).astype(np.float64)
        resistance[~np.isfinite(resistance)] = np.inf
        transform = ds.transform
        crs = ds.crs
        height, width = resistance.shape

    # Load path
    path = []
    with open(path_csv) as f:
        next(f)  # Skip header
        for line in f:
            row, col = map(int, line.strip().split(','))
            path.append((row, col))

    # Load metadata for start/goal
    with open(metadata_json) as f:
        metadata = json.load(f)

    # Create GridConfig
    cfg = GridConfig(
        resistance=resistance,
        height=height,
        width=width,
        cell_size=metadata["cell_size_m"],
        start=tuple(path[0]),
        goal=tuple(path[-1]),
    )

    print(f"Grid: {cfg.height}x{cfg.width}, cell={cfg.cell_size}m, CRS={crs}")
    print(f"Start: {cfg.start}, Goal: {cfg.goal}")
    print(f"Path cells (from CSV): {len(path)}")

    # Verify path cost
    computed_cost = path_cost(cfg, path)
    computed_length = path_physical_length(cfg, path)
    print(f"Computed cost: {computed_cost:.4f}")
    print(f"Computed length: {computed_length:.2f} m ({computed_length/1000:.3f} km)")

    # Run Dijkstra to verify
    result = dijkstra(cfg)
    print(f"\nDijkstra result:")
    print(f"  Reached: {result.reached}")
    print(f"  Path cells: {len(result.path)}")
    print(f"  Cost: {result.total_cost:.4f}")
    print(f"  Length: {result.physical_length_m:.2f} m ({result.physical_length_m/1000:.3f} km)")

    # Verify matches
    cost_match = abs(result.total_cost - computed_cost) < 0.01
    len_match = abs(result.physical_length_m - computed_length) < 1
    cells_match = len(result.path) == len(path)

    print(f"\nVerification:")
    print(f"  Cost matches CSV: {'OK' if cost_match else 'FAIL'}")
    print(f"  Length matches CSV: {'OK' if len_match else 'FAIL'}")
    print(f"  Cell count matches: {'OK' if cells_match else 'FAIL'}")

    # Load patches
    patches_df = pd.read_csv(patches_csv)
    print(f"\nHabitat patches: {len(patches_df)} total")
    print(f"  Tree cover: {len(patches_df[patches_df['Class Code'] == 10])}")
    print(f"  Shrubland: {len(patches_df[patches_df['Class Code'] == 20])}")
    print(f"  Grassland: {len(patches_df[patches_df['Class Code'] == 30])}")
    print(f"  Total habitat area: {patches_df['Area (ha)'].sum():.1f} ha")

    # Print top 5 largest patches
    print("\nTop 5 largest patches:")
    for _, row in patches_df.head(5).iterrows():
        print(f"  Patch {row['Patch ID']}: {row['Class Name']}, {row['Area (cells)']} cells ({row['Area (ha)']:.1f} ha)")

    all_ok = cost_match and len_match and cells_match and result.reached
    print(f"\n{'ALL VERIFICATIONS PASSED' if all_ok else 'SOME VERIFICATIONS FAILED'}")
    return all_ok


def run_habitat_analysis_synthetic() -> bool:
    """Run habitat patch identification on synthetic data."""
    print("=" * 60)
    print("HABITAT PATCH IDENTIFICATION (Synthetic)")
    print("=" * 60)

    paths = get_data_paths()
    raw_synthetic = paths["raw_synthetic"]
    processed_synthetic = paths["processed_synthetic"]

    with rasterio.open(str(raw_synthetic / "synthetic_landcover_5km.tif")) as ds:
        landcover = ds.read(1)
    with rasterio.open(str(raw_synthetic / "synthetic_resistance_5km.tif")) as ds:
        resistance = ds.read(1).astype(np.float64)
        resistance[~np.isfinite(resistance)] = np.inf

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

    print("\n[OK] habitat analysis passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ecological Corridor Optimizer - Least-cost corridor analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  synthetic   Run synthetic dataset regression test (baseline verification)
  habitat     Run habitat patch identification on synthetic data
  real        Verify real data pipeline outputs
  all         Run all tests

Examples:
  python -m corridor_optimizer.cli.main synthetic
  python -m corridor_optimizer.cli.main real
  python -m corridor_optimizer.cli.main all
        """
    )
    parser.add_argument(
        "command",
        choices=["synthetic", "habitat", "real", "all"],
        help="Command to run"
    )

    args = parser.parse_args()

    print("ECOLOGICAL CORRIDOR OPTIMIZER")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    success = True

    if args.command == "synthetic" or args.command == "all":
        success &= run_synthetic_regression()
        print()

    if args.command == "habitat" or args.command == "all":
        success &= run_habitat_analysis_synthetic()
        print()

    if args.command == "real" or args.command == "all":
        success &= run_real_data_verification()
        print()

    if success:
        print("ALL TASKS COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("SOME TASKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()