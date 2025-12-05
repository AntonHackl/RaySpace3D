#!/usr/bin/env bash

# Build RaySpace3D using conda

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Check if conda environment is active
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: No conda environment detected!"
    echo "Please activate the rayspace3d conda environment first:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate rayspace3d"
    exit 1
fi

BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "==================================="
echo "Building RaySpace3D"
echo "Using conda environment: $CONDA_PREFIX"
echo "Build type: $BUILD_TYPE"
echo "==================================="

# Set OptiX path
export OptiX_INSTALL_DIR="${OptiX_INSTALL_DIR:-/opt/optix}"
if [ ! -d "$OptiX_INSTALL_DIR/include" ]; then
    echo "Error: OptiX SDK not found at $OptiX_INSTALL_DIR"
    echo "Please set OptiX_INSTALL_DIR to your OptiX installation path"
    exit 1
fi

echo "Using OptiX SDK at: $OptiX_INSTALL_DIR"

# Use system compilers to avoid conda sysroot issues
export CC="gcc"
export CXX="g++"

# Use system CUDA or conda CUDA
if command -v nvcc &> /dev/null; then
    export CUDACXX=$(which nvcc)
    export CUDA_HOST_COMPILER="g++"
    echo "Using system CUDA: $CUDACXX"
elif [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDACXX="$CONDA_PREFIX/bin/nvcc"
    export CUDA_HOST_COMPILER="g++"
    echo "Using conda CUDA: $CUDACXX"
else
    echo "Error: nvcc not found. Please install CUDA toolkit"
    exit 1
fi

# Set up library and include paths
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

# Add system library paths for CUDA linking
export LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Don't use sysroot for linking to avoid library conflicts
unset LDFLAGS
unset CPPFLAGS
unset CFLAGS
unset CXXFLAGS


# Download tinyobjloader if missing
TINYOBJ_DIR="$PROJECT_DIR/external/tinyobjloader"
if [ ! -f "$TINYOBJ_DIR/tiny_obj_loader.h" ] || [ ! -f "$TINYOBJ_DIR/tiny_obj_loader.cc" ]; then
    echo "Downloading tinyobjloader..."
    mkdir -p "$TINYOBJ_DIR"
    wget -q -O "$TINYOBJ_DIR/tiny_obj_loader.h" https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.h
    wget -q -O "$TINYOBJ_DIR/tiny_obj_loader.cc" https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.cc
    echo "tinyobjloader downloaded to $TINYOBJ_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake..."
cmake ..

echo "Building..."
make -j$(nproc)

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo "Binaries: $BUILD_DIR/bin/"
echo "Run examples: $BUILD_DIR/bin/raytracer"
echo ""
