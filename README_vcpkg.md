# RaySpace3D

High-performance 3D spatial queries and ray tracing using OptiX and CGAL.

## Features
- Fast 3D point-in-mesh and ray tracing queries
- Batch processing of large datasets
- Modular C++/CUDA codebase
- Cross-platform (Linux, Windows)

## Requirements
- C++17 compiler
- CUDA Toolkit (for OptiX)
- CMake 3.18+
- [vcpkg](https://github.com/microsoft/vcpkg) (for dependency management)
- NVIDIA OptiX SDK (for ray tracing)

## Setup with vcpkg

1. **Install vcpkg** (if not already):
   ```bash
   git clone https://github.com/microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh
   ```

2. **Install dependencies** (optional - the build script will auto-install):
   ```bash
   # If you want to pre-install packages, run:
   /opt/vcpkg/vcpkg install --triplet x64-linux
   
   # Or if vcpkg is elsewhere:
   $VCPKG_ROOT/vcpkg install --triplet x64-linux
   ```
   
   Note: The CMake build will automatically install missing vcpkg packages when you run the build script.

3. **Quick build (recommended)** — use the included build script:
   ```bash
   # from the project root
   ./scripts/build_vcpkg.sh
   ```

   The script uses the following environment variables (optional):
   - `VCPKG_ROOT` : path to your vcpkg installation (if omitted the script will try common locations)
   - `VCPKG_TRIPLET` : vcpkg triplet (default `x64-linux`)
   - `BUILD_TYPE` : CMake build type (default `Release`)

4. **Run the executables:**
   Binaries will be in `build/bin/`.

## Notes
- Ensure the OptiX SDK is installed and available. You may need to set `OptiX_INCLUDE` and related variables in your environment or CMake cache.
- CUDA and OptiX are required for ray tracing features.
- For more details, see the source code and comments in `CMakeLists.txt`.

## Legacy/Alternative Build (Conda)
If you prefer Conda, see the CGAL baseline for an example environment setup.
