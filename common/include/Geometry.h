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

struct GeometryData {
    // CPU-side data (std::vectors for flexibility)
    // Use PinnedAllocator to allow direct DMA transfer from these vectors
    std::vector<float3, PinnedAllocator<float3>> vertices;
    std::vector<uint3, PinnedAllocator<uint3>> indices;
    std::vector<int, PinnedAllocator<int>> triangleToObject;
    size_t totalTriangles = 0;
};

struct PointData {
    // CPU-side data (std::vector for flexibility)
    // Use PinnedAllocator to allow direct DMA transfer from these vectors
    std::vector<float3, PinnedAllocator<float3>> positions;
    size_t numPoints = 0;
};
