#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include "Geometry.h"

namespace RaySpace {
namespace IO {

constexpr uint32_t BINARY_FILE_MAGIC = 0x52334442; // "R3DB"
constexpr uint32_t BINARY_FILE_VERSION = 1;

struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t numVertices;
    uint64_t numIndices;
    uint64_t numMappings;
    uint64_t totalTriangles;
    uint8_t hasGrid;
    uint8_t padding[7]; // Align to 8 bytes
};

struct GridParams {
    float minBound[3];
    float maxBound[3];
    uint32_t resolution[3];
    uint32_t padding; // Align
};

// Write geometry and grid data to a binary file
inline bool writeBinaryFile(const std::string& filename, const GeometryData& geometry) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    FileHeader header;
    header.magic = BINARY_FILE_MAGIC;
    header.version = BINARY_FILE_VERSION;
    header.numVertices = geometry.vertices.size();
    header.numIndices = geometry.indices.size();
    header.numMappings = geometry.triangleToObject.size();
    header.totalTriangles = geometry.totalTriangles;
    header.hasGrid = geometry.grid.hasGrid ? 1 : 0;
    
    // Write Header
    out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));

    // Write Main Data Arrays
    if (header.numVertices > 0)
        out.write(reinterpret_cast<const char*>(geometry.vertices.data()), header.numVertices * sizeof(float3));
    
    if (header.numIndices > 0)
        out.write(reinterpret_cast<const char*>(geometry.indices.data()), header.numIndices * sizeof(uint3));

    if (header.numMappings > 0)
        out.write(reinterpret_cast<const char*>(geometry.triangleToObject.data()), header.numMappings * sizeof(int));

    // Write Grid Data if present
    if (header.hasGrid) {
        GridParams gp;
        gp.minBound[0] = geometry.grid.minBound.x;
        gp.minBound[1] = geometry.grid.minBound.y;
        gp.minBound[2] = geometry.grid.minBound.z;
        gp.maxBound[0] = geometry.grid.maxBound.x;
        gp.maxBound[1] = geometry.grid.maxBound.y;
        gp.maxBound[2] = geometry.grid.maxBound.z;
        gp.resolution[0] = geometry.grid.resolution.x;
        gp.resolution[1] = geometry.grid.resolution.y;
        gp.resolution[2] = geometry.grid.resolution.z;

        out.write(reinterpret_cast<const char*>(&gp), sizeof(GridParams));

        size_t numCells = geometry.grid.cells.size();
        if (numCells != (size_t)gp.resolution[0] * gp.resolution[1] * gp.resolution[2]) {
            std::cerr << "Warning: Grid cell count mismatch in write!" << std::endl;
        }
        
        // Write cells directly
        if (numCells > 0)
            out.write(reinterpret_cast<const char*>(geometry.grid.cells.data()), numCells * sizeof(GridCell));
    }

    out.close();
    return true;
}

// Read geometry and grid data from a binary file
inline GeometryData readBinaryFile(const std::string& filename) {
    GeometryData geometry;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return geometry;
    }

    FileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(FileHeader));

    if (header.magic != BINARY_FILE_MAGIC) {
        std::cerr << "Error: Invalid file format (Magic mismatch). File: " << filename << std::endl;
         // Fallback or empty return? 
         // For now return empty, could also throw exception
        return geometry;
    }

    geometry.totalTriangles = header.totalTriangles;
    geometry.vertices.resize(header.numVertices);
    geometry.indices.resize(header.numIndices);
    geometry.triangleToObject.resize(header.numMappings);

    if (header.numVertices > 0)
        in.read(reinterpret_cast<char*>(geometry.vertices.data()), header.numVertices * sizeof(float3));

    if (header.numIndices > 0)
        in.read(reinterpret_cast<char*>(geometry.indices.data()), header.numIndices * sizeof(uint3));

    if (header.numMappings > 0)
        in.read(reinterpret_cast<char*>(geometry.triangleToObject.data()), header.numMappings * sizeof(int));

    if (header.hasGrid) {
        geometry.grid.hasGrid = true;
        GridParams gp;
        in.read(reinterpret_cast<char*>(&gp), sizeof(GridParams));

        geometry.grid.minBound = {gp.minBound[0], gp.minBound[1], gp.minBound[2]};
        geometry.grid.maxBound = {gp.maxBound[0], gp.maxBound[1], gp.maxBound[2]};
        geometry.grid.resolution = {gp.resolution[0], gp.resolution[1], gp.resolution[2]};

        size_t numCells = (size_t)gp.resolution[0] * gp.resolution[1] * gp.resolution[2];
        geometry.grid.cells.resize(numCells);
        
        if (numCells > 0)
            in.read(reinterpret_cast<char*>(geometry.grid.cells.data()), numCells * sizeof(GridCell));
    }

    in.close();
    return geometry;
}

} // namespace IO
} // namespace RaySpace
