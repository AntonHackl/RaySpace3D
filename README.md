# RaySpace3D

RaySpace3D uses NVIDIA OptiX hardware-accelerated ray tracing to perform spatial join operations on 3D mesh datasets. It supports point-in-mesh containment, mesh overlap (boundary intersection), full mesh intersection (overlap + containment), and selectivity estimation.

## Architecture

The project is split into two separately built components that share a common library:

```
RaySpace3D/
├── common/              # Shared headers and utilities
│   ├── include/         # Geometry types, binary I/O, pinned memory, grid cells
│   └── src/             # Timer, CGAL triangulation helpers
├── preprocess/          # Dataset preprocessing (CGAL 6.0, tinyobjloader, tdbase)
│   ├── src/             # Dataset loaders and preprocessing main
│   └── environment.yml
├── query/               # Query executables (OptiX 7.x, CUDA)
│   ├── src/
│   │   ├── applications/   # Main programs (one per query type)
│   │   ├── cuda/           # CUDA/OptiX device code (ray generation, hit shaders)
│   │   ├── geometry/       # Bounding box computation, GPU memory uploads
│   │   ├── optix/          # OptiX context, pipeline, acceleration structure wrappers
│   │   ├── raytracing/     # Host-side launch orchestration (RayLauncher, FilterRefine, etc.)
│   │   └── runtime/        # File I/O for geometry and point data
│   └── environment.yml
└── SELECTIVITY_ESTIMATION.md  # Detailed selectivity estimation methodology
```

The two components have separate conda environments because tdbase (used by the preprocessor) requires CGAL 6.0, which conflicts with the OptiX/CUDA build environment.

## Prerequisites

- **Linux** with NVIDIA GPU (compute capability >= 7.5)
- **NVIDIA OptiX SDK 7.5+** — set `OptiX_INSTALL_DIR` environment variable to the SDK root
- **CUDA Toolkit 12.x**
- **Conda** (Miniforge or Miniconda)

## Building

### Preprocess

```bash
cd preprocess
conda env create -f environment-linux.yml
conda activate rayspace3d_preprocess_linux
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Produces: `build/bin/preprocess_dataset`

### Query

```bash
cd query
conda env create -f environment-linux.yml
conda activate rayspace3d_query_linux
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Produces the following executables in `build/bin/`:

| Executable | Query Type |
|---|---|
| `raytracer` | Point-in-mesh containment |
| `raytracer_filter_refine` | Point-in-mesh with bounding-box pre-filter |
| `raytracer_mesh_overlap` | Mesh overlap join (boundary intersection) |
| `raytracer_mesh_intersection` | Full mesh intersection join (overlap + containment) |
| `raytracer_containment` | Mesh containment join (object-in-object) |
| `raytracer_overlap_estimated` | Mesh overlap with selectivity estimation (power-of-2 hash table) |
| `raytracer_overlap_direct_estimation` | Mesh overlap with direct selectivity estimation |
| `raytracer_intersection_estimated` | Mesh intersection with selectivity estimation |

## Workflow

### 1. Preprocess

Convert raw mesh files (`.obj` or `.dt`) into the binary format consumed by the query tools:

```bash
# Basic preprocessing
./preprocess_dataset --dataset input.obj --output-geometry output.pre

# With grid statistics for selectivity estimation
./preprocess_dataset --dataset input.obj --output-geometry output.pre \
    --generate-grid --grid-resolution 128
```

**Options:**

| Flag | Description |
|---|---|
| `--mode <wkt\|mesh\|dt>` | Input format (default: auto-detected from extension) |
| `--dataset <path>` | Path to input file |
| `--output-geometry <path>` | Output binary file |
| `--output-timing <path>` | Timing JSON output (default: `preprocessing_timing.json`) |
| `--shuffle` | Apply random translation to each object |
| `--generate-grid` | Compute grid statistics for selectivity estimation |
| `--grid-resolution <N>` | Grid cells per axis (default: 128) |
| `--world-size <S>` | Fixed world bounds; omit to auto-detect from data |

### 2. Query

All query tools read preprocessed `.pre` files and write timing results to JSON.

**Point-in-mesh containment:**
```bash
./raytracer --geometry-files mesh.pre --point-files points.wkt --timing-file timing.json
```

**Filter-refine containment** (first tests against bounding box, then exact geometry):
```bash
./raytracer_filter_refine --geometry mesh.pre --points points.wkt --timing timing.json
```

**Mesh overlap join** (finds object pairs whose boundaries intersect):
```bash
./raytracer_mesh_overlap --mesh1 meshA.pre --mesh2 meshB.pre --timing timing.json
```

**Full mesh intersection join** (overlap + containment fallback for fully contained objects):
```bash
./raytracer_mesh_intersection --mesh1 meshA.pre --mesh2 meshB.pre --timing timing.json
```

**With selectivity estimation** (pre-estimates result size to optimize hash table allocation):
```bash
./raytracer_overlap_estimated --mesh1 meshA.pre --mesh2 meshB.pre \
    --gamma 0.8 --epsilon 0.001 --timing timing.json
```

**Estimation-only mode** (print estimate without executing the full query):
```bash
./raytracer_overlap_estimated --mesh1 meshA.pre --mesh2 meshB.pre --estimate-only
```

## Query Approaches Explained

### Point-in-Mesh Containment (`raytracer`)
Casts one ray per query point against the mesh acceleration structure. A point is inside the mesh if it hits an odd number of triangles (ray-casting parity test).

### Filter-Refine (`raytracer_filter_refine`)
Two-phase approach: first traces rays against the mesh's axis-aligned bounding box to eliminate points that are clearly outside, then runs the exact containment test only on the surviving candidates.

### Mesh Overlap (`raytracer_mesh_overlap`)
For each triangle edge in Mesh A, casts a ray against Mesh B's acceleration structure (and vice versa). Any intersection identifies a pair of overlapping objects. Results are deduplicated using an on-GPU hash table, then a sort-based deduplication step merges the bidirectional results.

### Mesh Intersection (`raytracer_mesh_intersection`)
Extends mesh overlap with a containment fallback: if an object in Mesh A has no edge intersections with Mesh B, it may still be fully contained. The algorithm tests one representative vertex per non-intersecting object to check containment via ray-casting parity.

### Mesh Containment (`raytracer_containment`)
Two-phase pipeline with separate OptiX pipelines: Phase 1 casts edge rays bidirectionally to find boundary intersections. Phase 2 tests one vertex per B-object against A's acceleration structure for point-in-mesh containment. Objects that are fully inside another (no edge hits, positive containment) are reported.

### Selectivity Estimation (`raytracer_overlap_estimated`, `raytracer_intersection_estimated`)
Uses pre-computed grid statistics (object density, average size, volume ratio per cell) to probabilistically estimate the number of result pairs *before* executing the query. This estimate sizes the GPU hash table to avoid both over-allocation and costly resizing. See [SELECTIVITY_ESTIMATION.md](SELECTIVITY_ESTIMATION.md) for the full methodology.

## Binary File Format

The preprocessor writes a custom binary format (magic: `R3DB`) containing:

| Section | Content |
|---|---|
| Header | Magic, version, counts for vertices/indices/mappings, triangle count, grid flag |
| Vertices | `float3[]` — vertex positions |
| Indices | `uint3[]` — triangle vertex indices |
| TriangleToObject | `int[]` — maps each triangle to its parent object ID |
| Grid (optional) | Bounds, resolution, and per-cell statistics (`GridCell[]`) |
