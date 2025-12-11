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
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<int> triangleToObject;
    size_t totalTriangles = 0;
    
    // Pinned memory buffers for fast GPU transfer
    PinnedGeometryBuffers pinnedBuffers;
};

struct PointData {
    // CPU-side data (std::vector for flexibility)
    std::vector<float3> positions;
    size_t numPoints = 0;
    
    // Pinned memory buffer for fast GPU transfer
    PinnedPointBuffers pinnedBuffers;
};
