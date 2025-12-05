# RaySpace3D

High-performance 3D spatial queries and ray tracing using OptiX and CGAL.

## Features
- Fast 3D point-in-mesh and ray tracing queries
- Batch processing of large datasets
- Modular C++/CUDA codebase
- Cross-platform (Linux, Windows)

## Requirements
- CUDA Toolkit (system installation)
- NVIDIA OptiX SDK (for ray tracing)
- Conda (for dependency management)

## Setup with Conda

### 1. Install OptiX SDK
Ensure the NVIDIA OptiX SDK is installed. By default, the build script expects it at `/opt/optix`. If installed elsewhere, set the `OptiX_INSTALL_DIR` environment variable:
```bash
export OptiX_INSTALL_DIR=/path/to/optix
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate rayspace3d
```

The environment includes:
- GCC 11.4 compilers
- CGAL for computational geometry
- Boost libraries
- GMP, MPFR for arbitrary precision arithmetic
- Eigen for linear algebra
- System CUDA toolkit integration

### 3. Build the Project
```bash
bash scripts/build.sh
```

The build script will:
- Detect the conda environment
- Use system CUDA compiler (nvcc)
- Use system GCC/G++ compilers
- Download and build tinyobjloader locally
- Configure CMake with conda library paths
- Build all executables in parallel

### 4. Run the Executables
Binaries are located in `build/bin/`:

```bash
# Ray tracing queries
./build/bin/raytracer --help

# Filter-refine approach
./build/bin/raytracer_filter_refine --help

# Dataset preprocessing
./build/bin/preprocess_dataset --help

# Single polygon triangulation
./build/bin/triangulate_single_polygon --help
```

## Build Configuration

### Environment Variables
- `OptiX_INSTALL_DIR`: Path to OptiX SDK (default: `/opt/optix`)
- `BUILD_TYPE`: CMake build type (default: `Release`)

### Build Options
```bash
# Debug build
BUILD_TYPE=Debug bash scripts/build.sh

# Specify OptiX location
OptiX_INSTALL_DIR=/custom/path bash scripts/build.sh
```

## Dependencies

All C++ dependencies are managed through conda:
- **CGAL**: Computational geometry algorithms
- **Boost**: C++ libraries (filesystem, system)
- **GMP/MPFR**: Arbitrary precision arithmetic
- **Eigen**: Linear algebra
- **TinyObjLoader**: OBJ file parsing (downloaded automatically)

CUDA and OptiX must be installed separately on the system.

## Troubleshooting

### CUDA not found
Ensure CUDA is installed and `nvcc` is in your PATH:
```bash
which nvcc
nvcc --version
```

### OptiX not found
Set the `OptiX_INSTALL_DIR` environment variable or install OptiX to `/opt/optix`.

### Build errors
Clean the build directory and rebuild:
```bash
rm -rf build
bash scripts/build.sh
```

## Notes
- The build uses system compilers (gcc/g++) instead of conda compilers to avoid sysroot conflicts
- CUDA must be compatible with your GPU driver version
- OptiX SDK version should match your CUDA version requirements
