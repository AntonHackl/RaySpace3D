#!/usr/bin/env bash

# Build RaySpace3D using conda

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Activate rayspace3d environment if not already active
if [ -z "${CONDA_PREFIX:-}" ] || [[ "$CONDA_PREFIX" != *"rayspace3d"* ]]; then
    echo "Activating rayspace3d conda environment..."
    # Source conda (disable strict mode temporarily)
    set +u
    if [ -f "$HOME/conda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/conda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Error: Could not find conda installation"
        exit 1
    fi
    
    conda activate rayspace3d || {
        echo "Error: Failed to activate rayspace3d environment"
        echo "Please create it first:"
        echo "  conda env create -f environment.yml"
        exit 1
    }
    set -u
fi

set -euo pipefail

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

# Don't override compilers - let conda environment provide them if available
# This ensures proper library paths are used
# export CC="gcc"
# export CXX="g++"

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

# Don't override conda environment's library paths and compiler flags
# The conda environment already sets these correctly


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

# After running CMake inside the project environment, switch out of the
# rayspace3d environment so that the actual `make` step runs in `base`.
# This avoids linking against conda-provided sysroots during the compile/link
# phase while keeping the configure step (CMake) using the conda environment.
echo "Switching to 'base' conda environment for the build (make)..."
# Try to activate base; if that fails, at minimum deactivate current env.
# Disable strict mode for conda operations
set +u
if command -v conda &> /dev/null; then
    # Prefer activating 'base' explicitly; fall back to simple deactivate.
    conda deactivate || true
    if conda activate base 2>/dev/null; then
        echo "Activated 'base' environment"
    else
        echo "Could not activate 'base' (continuing with environment deactivated)"
        # Ensure we are at least deactivated
        conda deactivate || true
    fi
else
    echo "conda command not found; leaving current shell environment as-is"
fi
set -u

echo "Building..."
make -j$(nproc)

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo "Binaries: $BUILD_DIR/bin/"
echo "Run examples: $BUILD_DIR/bin/raytracer"
echo ""
