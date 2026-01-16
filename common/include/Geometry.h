#pragma once

#include <vector>

// PinnedMemory.h includes cuda_runtime.h, which defines float3/uint3
// So we must include it first to avoid redefinition conflicts
#include "PinnedMemory.h"

#ifdef INCLUDE_OPTIX
#include <optix.h>
// cuda_runtime.h already included via PinnedMemory.h
#endif

// No need for custom float3/uint3 definitions anymore
// They come from cuda_runtime.h via PinnedMemory.h

struct EulerHistogram {
    int nx = 0, ny = 0, nz = 0;
    float3 minBound = { 0,0,0 };
    float3 maxBound = { 0,0,0 };
    float3 cellSize = { 0,0,0 };

    // Flattened 3D grids
    // We can use standard vectors for these as we might not need them strictly on GPU in pinned memory 
    // immediately, or we can use pinned. Standard is fine for now as we transfer manually if needed or read/write on CPU.
    std::vector<int> v_counts;
    std::vector<int> e_counts;
    std::vector<int> f_counts;
    std::vector<int> object_counts;
};

struct GeometryData {
    // CPU-side data (std::vectors for flexibility)
    // Use PinnedAllocator to allow direct DMA transfer from these vectors
    std::vector<float3, PinnedAllocator<float3>> vertices;
    std::vector<uint3, PinnedAllocator<uint3>> indices;
    std::vector<int, PinnedAllocator<int>> triangleToObject;
    size_t totalTriangles = 0;

    EulerHistogram eulerHistogram;
};

struct PointData {
    // CPU-side data (std::vector for flexibility)
    // Use PinnedAllocator to allow direct DMA transfer from these vectors
    std::vector<float3, PinnedAllocator<float3>> positions;
    size_t numPoints = 0;
};
