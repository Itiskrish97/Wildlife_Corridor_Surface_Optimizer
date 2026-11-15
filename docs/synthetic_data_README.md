# Synthetic Ecological Corridor Landscape

This dataset is a deliberately constructed synthetic example for algorithm development.
It is NOT a real ESA WorldCover extract and does NOT represent actual land cover at Asola Bhatti,
Mangar Bani, or any other real location.

## Spatial dimensions
- Grid: 250 rows x 250 columns
- Cell size: 20 m x 20 m
- Model extent: 5,000 m x 5,000 m (5 km x 5 km)
- CRS tag: EPSG:32643, used as a projected coordinate reference for raster compatibility.
- IMPORTANT: the coordinate origin is arbitrary. The GeoTIFF is not georeferenced to a real site.

## Files
- synthetic_ecological_landscape_map.png: map with scale bar, north arrow, grid, legend.
- synthetic_landcover_5km.tif: integer land-cover codes.
- synthetic_resistance_5km.tif: resistance values.
- synthetic_landcover_grid.csv: 250x250 land-cover code matrix.
- synthetic_resistance_grid.csv: 250x250 resistance matrix (inf = blocked).
- class_resistance_legend.csv: code, category, resistance, colour and treatment.

## Class codes
1 dense habitat/tree cover; 2 shrubland; 3 grassland; 4 cropland;
5 barren; 6 built-up; 7 road; 8 water.

## Illustrative resistance
1: 1; 2: 2; 3: 3; 4: 5; 5: 7; 6: blocked/infinite; 7: 12; 8: 8.
These are experimental values only, not scientifically validated species parameters.

## Suggested algorithm inputs
- Habitat A seed cell: row 116, column 42.
- Habitat B seed cell: row 133, column 208.
- Grid movement: 8-neighbour recommended; use proper diagonal movement costs.
- Use Dijkstra as baseline and A* with an admissible heuristic.
- Treat built-up cells as blocked. Treat roads and water according to the chosen scenario.
