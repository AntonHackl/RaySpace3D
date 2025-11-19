# Grid Query Benchmark Script

This script performs a spatial query benchmark by translating query geometry across a 3D grid and measuring raytracer performance at each position.

## Features

- **3x3x3 Grid (or custom)**: Translates query geometry to 27 different positions (configurable)
- **Multiple Runs**: Executes raytracer multiple times per position for statistical reliability
- **Performance Analysis**: Computes mean, std dev, min, max query times
- **3D Visualization**: Generates a 3D plot showing query performance across space
- **JSON Output**: Saves all results in structured JSON format

## Requirements

```bash
pip install numpy matplotlib
# Optional for interactive 3D plots:
pip install plotly
```

## Usage

### Basic Example

```bash
python scripts/grid_query_benchmark.py \
    --geometry generated_files/query.txt \
    --points generated_files/shape_net_points_1000000.wkt \
    --bbox generated_files/shape_net_bbox.json \
    --output results/grid_benchmark.json \
    --runs 5
```

### Custom Grid Size

```bash
python scripts/grid_query_benchmark.py \
    --geometry generated_files/query.txt \
    --points generated_files/shape_net_points_1000000.wkt \
    --bbox generated_files/shape_net_bbox.json \
    --output results/grid_benchmark_5x5x5.json \
    --runs 10 \
    --grid-size 5 5 5
```

### With Custom Raytracer Path

```bash
python scripts/grid_query_benchmark.py \
    --geometry generated_files/query.txt \
    --points generated_files/shape_net_points_1000000.wkt \
    --bbox generated_files/shape_net_bbox.json \
    --raytracer build/bin/Release/raytracer.exe \
    --output results/grid_benchmark.json
```

## How It Works

1. **Load Data**: Reads query geometry, point dataset, and bounding box
2. **Generate Grid**: Creates a 3D grid of positions within the bounding box
3. **For Each Grid Position**:
   - Translates query geometry to grid cell center
   - Saves translated geometry to temporary file
   - Runs raytracer with specified number of runs
   - Extracts query timing from JSON output
4. **Analysis**: Computes statistics across all grid positions
5. **Visualization**: Creates 3D scatter plot with color-coded performance
6. **Output**: Saves results to JSON and visualization to PNG

## Output Files

### JSON Output (`--output`)
Contains:
- Configuration (files, grid size, runs)
- Statistics (mean, std, min, max query times)
- Detailed results for each grid position with timing data

### Plot Output (`--plot` or auto-generated)
- Static PNG: 3D scatter plot with viridis colormap
- Interactive HTML (if plotly installed): Rotatable 3D visualization

## Example Output

```
Query Time Statistics (27 successful queries):
  Mean:     12.45 ms
  Std Dev:  3.21 ms
  Min:      8.32 ms
  Max:      18.76 ms
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--geometry` | Yes | - | Preprocessed query geometry file |
| `--points` | Yes | - | Point dataset in WKT format |
| `--bbox` | Yes | - | Bounding box JSON file |
| `--raytracer` | No | `build/bin/Debug/raytracer.exe` | Path to raytracer executable |
| `--output` | No | `results/grid_benchmark.json` | Output JSON file |
| `--plot` | No | (auto) | Output PNG file (defaults to output with .png) |
| `--runs` | No | 5 | Number of runs per grid position |
| `--grid-size` | No | 3 3 3 | Grid dimensions (X Y Z) |
| `--temp-dir` | No | (system temp) | Directory for temporary files |

## Notes

- Each grid cell uses the **center** of the cell as the translation target
- Query geometry is translated but **normals are not** (they're directional vectors)
- The script creates temporary files for each translated geometry
- Timeout per raytracer execution: 600 seconds (10 minutes)
- Failed queries are excluded from statistics but logged in results

## Troubleshooting

### "Raytracer executable not found"
Make sure you've built the project:
```bash
cmake --build build --config Debug
```

### "Geometry file not found"
Preprocess your dataset first:
```bash
build/bin/Debug/preprocess_dataset.exe --dataset <path> --output-geometry query.txt
```

### Script runs but no visualization
Install matplotlib:
```bash
pip install matplotlib
```

### Want interactive 3D plots?
Install plotly:
```bash
pip install plotly
```
