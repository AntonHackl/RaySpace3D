# RaySpace3D

High-performance 3D spatial queries and ray tracing using OptiX and CGAL.

## Features
- Fast 3D point-in-mesh and ray tracing queries
- Batch processing of large datasets
- Modular C++/CUDA codebase
- Cross-platform (Linux, Windows)

## Requirements
- CUDA Toolkit (conda installation on Windows, system or conda on Linux)
- NVIDIA OptiX SDK (for ray tracing)
- Conda (for dependency management)

## Setup with Conda

### 1. Install OptiX SDK

**Linux:**
Ensure the NVIDIA OptiX SDK is installed. By default, the build script expects it at `/opt/optix`. If installed elsewhere, set the `OptiX_INSTALL_DIR` environment variable:
```bash
export OptiX_INSTALL_DIR=/path/to/optix
```

**Windows:**
The OptiX SDK should be installed at `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0` (or set `OptiX_INSTALL_DIR` environment variable).

### 2. Create Conda Environment

**Windows:**
```bash
conda env create -f environment.yml
conda activate rayspace3d
```

**Linux:**
```bash
conda env create -f environment-linux.yml
conda activate rayspace3d
```

The environment includes:
- **Linux**: GCC 11.4 compilers, CUDA toolkit (conda or system)
- **Windows**: CUDA toolkit from conda (nvidia channel)
- CGAL for computational geometry
- Boost libraries
- GMP, MPFR for arbitrary precision arithmetic
- Eigen for linear algebra

**Note:** Linux and Windows use separate environment files because Linux requires platform-specific compiler packages (`gcc_linux-64`, `gxx_linux-64`, `sysroot_linux-64`) that are not available on Windows.

### 3. Build the Project

**Linux:**
```bash
bash scripts/build.sh
```

**Windows:**
```powershell
.\scripts\build.ps1
```

**Note for Windows:** The build script automatically finds and initializes conda, so you don't need to use Anaconda Prompt. If you prefer to use VS Code tasks, they will also work from regular PowerShell (the tasks use a helper script that finds conda automatically).

The build scripts will:
- Detect and activate the conda environment
- **Windows**: Use conda CUDA toolkit (required)
- **Linux**: Use conda or system CUDA compiler (nvcc)
- Download and build tinyobjloader locally
- Configure CMake with conda library paths
- Build all executables in parallel

### 4. Run the Executables
Binaries are located in `build/bin/` (Linux) or `build/bin/Debug/` / `build/bin/Release/` (Windows):

**Linux:**
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

**Windows:**
```powershell
# Ray tracing queries
.\build\bin\Debug\raytracer.exe --help

# Filter-refine approach
.\build\bin\Debug\raytracer_filter_refine.exe --help

# Dataset preprocessing
.\build\bin\Debug\preprocess_dataset.exe --help

# Single polygon triangulation
.\build\bin\Debug\triangulate_single_polygon.exe --help
```

## Build Configuration

### Environment Variables
- `OptiX_INSTALL_DIR`: Path to OptiX SDK (default: `/opt/optix`)
- `BUILD_TYPE`: CMake build type (default: `Release`)

### Build Options

**Linux:**
```bash
# Debug build
BUILD_TYPE=Debug bash scripts/build.sh

# Specify OptiX location
OptiX_INSTALL_DIR=/custom/path bash scripts/build.sh
```

**Windows:**
```powershell
# Debug build
$env:BUILD_TYPE="Debug"; .\scripts\build.ps1

# Specify OptiX location
$env:OptiX_INSTALL_DIR="C:\Custom\Path\To\OptiX"; .\scripts\build.ps1
```

## Dependencies

All C++ dependencies are managed through conda:
- **CGAL**: Computational geometry algorithms
- **Boost**: C++ libraries (filesystem, system)
- **GMP/MPFR**: Arbitrary precision arithmetic
- **Eigen**: Linear algebra
- **TinyObjLoader**: OBJ file parsing (downloaded automatically)

- **Windows**: CUDA toolkit is provided via conda (from nvidia channel). OptiX SDK must be installed separately on the system.
- **Linux**: CUDA toolkit can be provided via conda or system installation. OptiX SDK must be installed separately on the system.

## Troubleshooting

### CUDA not found

**Windows:**
Ensure conda CUDA toolkit is installed in the environment:
```powershell
conda activate rayspace3d
conda list cuda-toolkit
nvcc --version
```

If not installed:
```powershell
conda install -c nvidia cuda-toolkit=12.8
```

**Linux:**
Ensure CUDA is installed and `nvcc` is in your PATH:
```bash
which nvcc
nvcc --version
```

### OptiX not found

**Linux:**
Set the `OptiX_INSTALL_DIR` environment variable or install OptiX to `/opt/optix`.

**Windows:**
Set the `OptiX_INSTALL_DIR` environment variable or install OptiX to `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0`.

### Build errors
Clean the build directory and rebuild:

**Linux:**
```bash
rm -rf build
bash scripts/build.sh
```

**Windows:**
```powershell
Remove-Item -Recurse -Force build
.\scripts\build.ps1
```

## Notes
- **Linux**: The build uses system compilers (gcc/g++) instead of conda compilers to avoid sysroot conflicts
- **Windows**: The build uses conda CUDA toolkit and MSVC compiler (from Visual Studio)
- **Windows conda setup**: The build scripts automatically find conda.exe even if conda is not in your PowerShell PATH. You don't need to use Anaconda Prompt - regular PowerShell works fine.
- CUDA must be compatible with your GPU driver version
- OptiX SDK version should match your CUDA version requirements
- On Windows, conda environment must remain active throughout the build process (unlike Linux where it's deactivated after CMake configure)
