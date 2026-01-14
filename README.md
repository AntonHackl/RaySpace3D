# RaySpace3D - Restructured Project

This project has been split into two separate components with their own environments:

## Project Structure

```
RaySpace3D/
├── preprocess/          # Dataset preprocessing (uses tdbase, CGAL 6.0)
│   ├── CMakeLists.txt
│   ├── environment.yml
│   ├── src/
│   └── bin/
├── query/               # Query execution with OptiX (uses CGAL 5.6)
│   ├── CMakeLists.txt
│   ├── environment.yml
│   ├── src/
│   └── bin/
└── common/             # Shared code between both
    ├── CMakeLists.txt
    ├── include/
    └── src/
```

## Setup

### 1. Preprocessing Tool

The preprocessing tool converts datasets (.obj, .dt) into the binary format used by the query engine.

**Create environment:**
```bash
cd preprocess
conda env create -f environment.yml
conda activate rayspace3d_preprocess
```

**Build:**
```bash
mkdir -p build && cd build
cmake ..
make -j8
```

**Usage:**
```bash
./bin/preprocess_dataset <input_file> <output_file>
```

### 2. Query Tool

The query tool performs spatial intersection queries using OptiX ray tracing.

**Create environment:**
```bash
cd query
conda env create -f environment.yml
conda activate rayspace3d_query
```

**Build:**
```bash
mkdir -p build && cd build
cmake ..
make -j8
```

**Usage:**
```bash
./bin/raytracer <args>
./bin/raytracer_mesh_overlap <args>
```

## Why Two Separate Projects?

- **Dependency Conflicts**: tdbase requires CGAL 6.0 which has incompatible boost requirements with the existing OptiX setup
- **Modularity**: Preprocessing is a one-time operation, queries are runtime operations
- **Environment Isolation**: Each tool can have its optimal dependency versions

## Workflow

1. **Preprocess** your datasets once using the preprocessing tool
2. **Query** the preprocessed data using the query tools as many times as needed


## Preprocessing file format
The preprocessing tool converts datasets (.obj, .dt) into a text format used for the query engine. The format is defined as follows:

```
vertices: ... (each vertex are three float values, the position of the vertex)
faces: ... (each face is three int values, a triangle)
triangleToObject: ... (each triangle is one int value, the index of the object it belongs to)
triangle_count: ... (one integer, the total number of triangles)