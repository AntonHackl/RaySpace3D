#pragma once

#include <vector>

#ifdef INCLUDE_OPTIX
#include <optix.h>
#include <cuda_runtime.h>
#else
// Define basic types for non-OptiX builds
struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
struct uint3 {
    unsigned int x, y, z;
    uint3() : x(0), y(0), z(0) {}
    uint3(unsigned int x_, unsigned int y_, unsigned int z_) : x(x_), y(y_), z(z_) {}
};
#endif

struct GeometryData {
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<uint3> indices;
    std::vector<int> triangleToObject;
    size_t totalTriangles = 0;
};

struct PointData {
    std::vector<float3> positions;
    size_t numPoints = 0;
};
