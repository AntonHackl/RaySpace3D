#!/usr/bin/env bash

# Build RaySpace3D using vcpkg

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Try to locate vcpkg
if [ -z "${VCPKG_ROOT:-}" ]; then
    if [ -f "$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" ]; then
        VCPKG_ROOT="$HOME/vcpkg"
    elif [ -f "/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" ]; then
        VCPKG_ROOT="/opt/vcpkg"
    elif [ -f "$HOME/.vcpkg/scripts/buildsystems/vcpkg.cmake" ]; then
        VCPKG_ROOT="$HOME/.vcpkg"
    else
        echo "Error: VCPKG_ROOT not set and vcpkg not found in common locations"
        echo "Set VCPKG_ROOT or install vcpkg into one of: $PROJECT_DIR/vcpkg, $PROJECT_DIR/../vcpkg, $HOME/vcpkg, /opt/vcpkg"
        echo "Example: export VCPKG_ROOT=~/vcpkg"
        exit 1
    fi
fi

VCPKG_TOOLCHAIN="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

if [ ! -f "$VCPKG_TOOLCHAIN" ]; then
    echo "Error: vcpkg toolchain file not found at $VCPKG_TOOLCHAIN"
    echo "Please check your VCPKG_ROOT setting: $VCPKG_ROOT"
    exit 1
fi

VCPKG_TRIPLET="${VCPKG_TRIPLET:-x64-linux}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "==================================="
echo "Building RaySpace3D"
echo "Using vcpkg at: $VCPKG_ROOT (triplet: $VCPKG_TRIPLET)"
echo "Build type: $BUILD_TYPE"
echo "==================================="

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Running CMake with vcpkg toolchain..."
cmake -DCMAKE_TOOLCHAIN_FILE="$VCPKG_TOOLCHAIN" -DVCPKG_TARGET_TRIPLET="$VCPKG_TRIPLET" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..

echo "Building..."
make -j$(nproc)

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo "Binaries: $BUILD_DIR/bin/"
echo "Run examples: $BUILD_DIR/bin/raytracer"
echo ""
echo "If you need to install vcpkg packages run:"
echo "  \$VCPKG_ROOT/vcpkg install --triplet $VCPKG_TRIPLET"
echo ""
