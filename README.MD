# Ecological Corridor Optimizer

**Least-cost ecological corridor identification using Dijkstra's algorithm on ESA WorldCover 2021 resistance surfaces.**

A Python toolkit for modeling landscape resistance from satellite land-cover data and computing optimal wildlife movement corridors between fragmented habitat patches.

---

## Overview

This project processes **ESA WorldCover 2021** land-cover data to:
1. **Model landscape resistance** — Convert land-cover classes to species-specific movement costs
2. **Identify habitat patches** — Find contiguous habitat fragments via connected-component analysis
3. **Compute least-cost corridors** — Run Dijkstra's algorithm on an 8-directional grid graph with symmetric edge costs and corner-cutting prevention

**Use case**: Evidence-based ecological connectivity planning for conservation, infrastructure mitigation, and species recovery.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Resistance Modeling** | Configurable CSV mapping ESA WorldCover classes → resistance values |
| **Habitat Identification** | Per-class connected-component labeling (8-connectivity) with size filtering |
| **Least-Cost Path Analysis** | 8-directional Dijkstra with symmetric costs, corner-cutting prevention |
| **Symmetric Edge Costs** | `cost(A→B) = distance × (res[A] + res[B]) / 2` — undirected graph |
| **Corner-Cutting Prevention** | Diagonal moves blocked if either orthogonal neighbor is impassable |
| **Verification Suite** | Baseline regression tests on synthetic + real data |
| **Modern Packaging** | `pyproject.toml`, editable install, CLI entry point |

---

## Project Structure

```
Wildlife_Corridor_Surface_Optimizer/
├── src/
│   └── corridor_optimizer/           # Python package
│       ├── __init__.py               # Package metadata
│       ├── algorithms/               # Core algorithmic modules
│       │   ├── __init__.py           # Public API exports
│       │   ├── grid_graph.py         # 8-directional grid graph
│       │   ├── dijkstra.py           # Dijkstra's algorithm
│       │   └── habitat.py            # Habitat patch identification
│       └── cli/
│           └── main.py               # CLI entry point
├── tests/
│   └── test_algorithms.py            # Unit tests (16 tests)
├── data/
│   ├── raw/                          # Original input data (immutable)
│   │   ├── real/                     # ESA WorldCover GeoTIFFs (Asola-Bhatti)
│   │   └── synthetic/                # Synthetic 5×5 km test landscape
│   └── processed/                    # Derived/intermediate data
│       ├── real/                     # Processed real data
│       └── synthetic/                # CSV grids, resistance legends
├── config/
│   └── resistance_model.csv          # ESA class → resistance mapping
├── outputs/
│   └── figures/                      # Generated visualizations
├── docs/
│   └── synthetic_data_README.md      # Synthetic dataset documentation
├── pyproject.toml                    # Modern Python packaging
├── requirements.txt                  # Dependencies
└── README.MD                         # This file
```

---

## Installation

```bash
# Clone and enter project
cd Wildlife_Corridor_Surface_Optimizer

# Install in development mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Requirements**: Python ≥3.10, numpy, scipy, pandas, rasterio, matplotlib, pillow

---

## Quick Start

```bash
# Run synthetic baseline regression test
python -m corridor_optimizer.cli.main synthetic

# Run habitat patch identification on synthetic data
python -m corridor_optimizer.cli.main habitat

# Verify real data pipeline outputs
python -m corridor_optimizer.cli.main real

# Run all verification tests
python -m corridor_optimizer.cli.main all

# Or use the installed entry point
corridor-optimizer synthetic
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `synthetic` | Run synthetic dataset regression test (validates algorithm correctness) |
| `habitat` | Run habitat patch identification on synthetic data |
| `real` | Verify real data pipeline outputs (loads pre-computed results) |
| `all` | Run all verification tests sequentially |

---

## Using as a Library

```python
from corridor_optimizer.algorithms import (
    load_grid,           # Load resistance GeoTIFF → GridConfig
    dijkstra,            # Compute least-cost path
    GridConfig,          # Grid configuration dataclass
    DijkstraResult,      # Result dataclass
    identify_habitat_patches,  # Find habitat patches
    filter_patches_by_distance,  # Find candidate patch pairs
    path_cost,           # Compute path resistance cost
    path_physical_length,# Compute path length in metres
)

# 1. Load resistance surface
cfg = load_grid("data/raw/synthetic/synthetic_resistance_5km.tif",
                start=(116, 42), goal=(133, 208))

# 2. Run Dijkstra
result = dijkstra(cfg)

# 3. Access results
print(f"Path: {len(result.path)} cells")
print(f"Cost: {result.total_cost:.4f}")
print(f"Length: {result.physical_length_m/1000:.3f} km")
print(f"Reached: {result.reached}")

# 4. Habitat analysis
from corridor_optimizer.algorithms import identify_habitat_patches
import rasterio

with rasterio.open("landcover.tif") as ds:
    landcover = ds.read(1)
with rasterio.open("resistance.tif") as ds:
    resistance = ds.read(1).astype(np.float64)
    resistance[~np.isfinite(resistance)] = np.inf

result = identify_habitat_patches(
    landcover=landcover,
    resistance=resistance,
    cell_size=20.0,
    eligible_classes=[10, 20, 30],  # ESA codes: Tree, Shrub, Grass
    class_names={10: "Tree cover", 20: "Shrubland", 30: "Grassland"},
    min_patch_cells=50,
)

print(f"Found {len(result.patches)} patches")
print(f"Total habitat: {result.total_habitat_hectares:.1f} ha")
```

---

## Core Algorithms

### 1. Grid Graph (`grid_graph.py`)

**8-directional Moore neighbourhood** with:
- **Orthogonal moves** (N, E, S, W): distance = 1.0 × cell_size
- **Diagonal moves** (NE, SE, SW, NW): distance = √2 × cell_size
- **Symmetric edge costs**: `cost = distance × (res[source] + res[target]) / 2`
- **Corner-cutting prevention**: Diagonal blocked if either adjacent orthogonal cell is impassable

```
Direction indices:
    0: N   (-1,  0)  dist=1.0
    1: NE  (-1, +1)  dist=√2
    2: E   ( 0, +1)  dist=1.0
    3: SE  (+1, +1)  dist=√2
    4: S   (+1,  0)  dist=1.0
    5: SW  (+1, -1)  dist=√2
    6: W   ( 0, -1)  dist=1.0
    7: NW  (-1, -1)  dist=√2
```

**Corner-check mapping**:
- NE (1) checks N (0) and E (2)
- SE (3) checks E (2) and S (4)
- SW (5) checks S (4) and W (6)
- NW (7) checks W (6) and N (0)

### 2. Dijkstra's Algorithm (`dijkstra.py`)

Standard implementation with:
- Binary heap priority queue (`heapq`)
- Early termination on goal
- Visited set for finalized nodes
- Path reconstruction via `came_from` dict
- Returns `DijkstraResult` with path, cost, length, explored nodes, timing

### 3. Habitat Patch Identification (`habitat.py`)

**Per-class connected-component analysis**:
- Uses `scipy.ndimage.label` with 8-connectivity (3×3 structure)
- Processes each eligible land-cover class independently
- Filters patches below `min_patch_cells` threshold
- Computes metrics: area (cells/hectares), centroid, bbox, perimeter, compactness
- Returns patches sorted by area (descending)

**Compactness metric**: `4π × Area / Perimeter²` (1.0 = circle, lower = irregular)

**Candidate pair filtering**: `filter_patches_by_distance(patches, min_sep, max_sep)` — finds patch pairs within distance range, sorted by combined area.

---

## Data Sources

### Synthetic Dataset (Development/Testing)
| Property | Value |
|----------|-------|
| Grid | 250 × 250 cells |
| Resolution | 20 m |
| Extent | 5 km × 5 km |
| CRS | EPSG:32643 (arbitrary origin) |
| Classes | 8 synthetic classes |
| Start | (116, 42) |
| Goal | (133, 208) |

**Synthetic resistance values**:
| Code | Class | Resistance | Treatment |
|------|-------|------------|-----------|
| 1 | Dense habitat | 1.0 | Traversable |
| 2 | Shrubland | 2.0 | Traversable |
| 3 | Grassland | 3.0 | Traversable |
| 4 | Cropland | 5.0 | Traversable |
| 5 | Barren | 7.0 | Traversable |
| 6 | Built-up | ∞ | Blocked |
| 7 | Road | 12.0 | High penalty |
| 8 | Water | 8.0 | Species-dependent |

### Real Data (Asola-Bhatti / Mangar Bani, India)
| Property | Value |
|----------|-------|
| Source | ESA WorldCover 2021 v200, tile N27E075 |
| AOI | 76.0–76.5°E, 27.0–27.5°N |
| Grid | 2795 × 2503 cells |
| Resolution | 20 m |
| Extent | ~56 × 50 km |
| CRS | EPSG:32643 (UTM Zone 43N) |
| Habitat classes | 10, 20, 30, 90, 95 |
| Total habitat | 103,201 ha |

**Real resistance model** (`config/resistance_model.csv`):
| ESA Code | Class | Resistance | Traversable |
|----------|-------|------------|-------------|
| 10 | Tree cover | 1.0 | ✓ |
| 20 | Shrubland | 2.0 | ✓ |
| 30 | Grassland | 3.0 | ✓ |
| 40 | Cropland | 5.0 | ✓ |
| 50 | Built-up | 999999 (∞) | ✗ |
| 60 | Bare/sparse | 7.0 | ✓ |
| 70 | Snow/ice | 999999 (∞) | ✗ |
| 80 | Water bodies | 8.0 | ✓ |
| 90 | Herbaceous wetland | 4.0 | ✓ |
| 95 | Mangroves | 3.0 | ✓ |
| 100 | Moss/lichen | 6.0 | ✓ |

> **Note**: Resistance values are *illustrative* — calibrate for target species using literature or expert input.

---

## Verified Metrics

### Synthetic Baseline (Regression Test)
| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Total resistance cost | 421.4874 | ±0.01 |
| Physical path length | 4,381.0 m | ±1 m |
| Path cells | 172 | Exact |

### Real Data Summary
| Metric | Value |
|--------|-------|
| Grid shape | 2795 × 2503 |
| Cell size | 20 m |
| Total habitat area | 103,201 ha |
| Habitat patches | 50 (top 50 by area) |
| Top classes | Grassland, Tree cover, Shrubland |
| Largest patch | 541,556 cells (21,662 ha) Grassland |
| Path length (CSV) | ~17.5 km |

---

## Resistance Model Configuration

Edit `config/resistance_model.csv` to customize for your species:

```csv
class_code,class_name,resistance,traversable,notes
10,Tree cover,1.0,1,"Lowest resistance - prime habitat"
20,Shrubland,2.0,1,"Good habitat"
30,Grassland,3.0,1,"Suitable for many species"
40,Cropland,5.0,1,"Fragmented, higher resistance"
50,Built-up,999999,0,"Effectively impassable"
60,Bare/sparse,7.0,1,"High resistance"
70,Snow/ice,999999,0,"Blocked"
80,Water,8.0,1,"Species-dependent"
90,Wetland,4.0,1,"Moderate resistance"
95,Mangroves,3.0,1,"Coastal habitat"
100,Moss/lichen,6.0,1,"Sparse vegetation"
```

- **resistance**: Higher = more costly to traverse
- **traversable**: 0 = blocked (infinite resistance), 1 = traversable
- Infinite resistance cells are excluded from the graph entirely

---

## Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_algorithms.py::TestGridGraph -v
python -m pytest tests/test_algorithms.py::TestDijkstra -v
python -m pytest tests/test_algorithms.py::TestHabitat -v

# Run with coverage
python -m pytest tests/ --cov=corridor_optimizer --cov-report=term-missing
```

**Test coverage**: 16 unit tests covering:
- Grid graph: directions, corner-cutting, neighbours, path cost, length
- Dijkstra: simple path, blocked goal, obstacle avoidance
- Habitat: patch identification, min-size filter, per-class separation, pair filtering

---

## Development

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Run all checks
black src/ tests/ && ruff check src/ tests/ && mypy src/ && python -m pytest tests/ -v
```

---

## Extending the Pipeline

### Add New Algorithms
1. Create `src/corridor_optimizer/algorithms/new_algo.py`
2. Export in `src/corridor_optimizer/algorithms/__init__.py`
3. Add tests in `tests/test_algorithms.py`

### A* Search (Planned)
```python
# Heuristic: Euclidean distance × min_resistance
def heuristic(a, b, min_res=1.0):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * min_res
```

### Multi-Species Corridor Overlay
```python
# Compute corridors for multiple species, then combine
corridors = [dijkstra(cfg_for_species(s)) for s in species]
# Union/intersection for multi-species connectivity
```

### Connectivity Metrics
- Integral Index of Connectivity (IIC)
- Probability of Connectivity (PC)
- Betweenness centrality on corridor network

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/figures/dijkstra_result.png` | Synthetic path visualization |
| `outputs/figures/real_data_overview.png` | Real data landcover overview |
| `outputs/figures/real_data_dijkstra_path.png` | Real data least-cost path |
| `data/processed/real/dijkstra_path.csv` | Path coordinates (row, col) |
| `data/processed/real/candidate_habitat_patches.csv` | Patch metrics |

---

## License

This project is for research and educational purposes. ESA WorldCover data © ESA 2021.

---

## Data Citation

**ESA WorldCover 2021 v200** (used in this project):

> Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., Carter, S., Herold, M., Li, Linlin, Tsendbazar, N.E., Ramoino, F., Arino, O., 2021. *ESA WorldCover 10 m 2020 v100*. https://doi.org/10.5281/zenodo.5571936

**Note**: This project uses the v200 (2021) release. The citation above is for v100 (2020); update the version/URL accordingly when citing.

---

## Software Citation

If you use this toolkit in research, please cite:
```
Ecological Corridor Optimizer — Least-cost corridor analysis using Dijkstra's algorithm
on ESA WorldCover 2021 resistance surfaces.
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: corridor_optimizer` | Run `pip install -e .` from project root |
| `rasterio` import error | Ensure GDAL is installed: `conda install -c conda-forge gdal` or use `pip install rasterio` with system GDAL |
| Memory error on real data | Real grid is 2795×2503 (~7M cells). Use smaller AOI or increase RAM |
| Path not found | Check start/goal are in traversable cells; verify resistance model |
| Different path than CSV | Real data verification loads pre-computed CSV; Dijkstra may find equally optimal alternative paths |

---

## Roadmap

- [ ] A* algorithm with admissible heuristic
- [ ] Multi-species corridor overlay
- [ ] Connectivity metrics (IIC, PC, betweenness)
- [ ] Uncertainty quantification (resistance sensitivity)
- [ ] Web visualization (Streamlit/Folium)
- [ ] GBIF species occurrence integration
- [ ] Dynamic landscape scenarios (future land-use)

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/xyz`)
3. Add tests for new functionality
4. Run full test suite (`pytest tests/ -v`)
5. Format/lint (`black src/ tests/ && ruff check src/ tests/`)
6. Submit PR with clear description

---

**Maintainer**: Krish Singh  
**Project**: Wildlife Corridor Surface Optimizer  
**Last Updated**: 2025-11-15