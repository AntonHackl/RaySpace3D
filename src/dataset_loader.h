#pragma once

#include <string>
#include <vector>

#include "triangulation.h" // For PolygonWithHoles definition

#ifdef INCLUDE_OPTIX
#include <optix.h>
#include <cuda_runtime.h>
#endif

#ifndef INCLUDE_OPTIX
// Define basic types for non-OptiX builds
struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct uint3 {
    unsigned int x, y, z;
    uint3() : x(0), y(0), z(0) {}
    uint3(unsigned int x, unsigned int y, unsigned int z) : x(x), y(y), z(z) {}
};
#endif

struct GeometryData {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<int> triangleToPolygon;
    size_t totalTriangles = 0;
};

struct PointData {
    std::vector<float3> positions;  // Ray origins (x, y, -1)
    size_t numPoints = 0;
};

// New split functions for dataset processing
std::vector<PolygonWithHoles> loadPolygonsWithHoles(const std::string& datasetPath);
GeometryData triangulatePolygons(const std::vector<PolygonWithHoles>& polygons);

// Backwards compatible wrapper
GeometryData loadDatasetGeometry(const std::string& datasetPath);

GeometryData loadGeometryFromFile(const std::string& geometryFilePath);
PointData loadPointDataset(const std::string& pointDatasetPath);