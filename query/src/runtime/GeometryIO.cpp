#include "GeometryIO.h"

#include <iostream>
#include <fstream>
#include <sstream>

GeometryData loadGeometryFromFile(const std::string& geometryFilePath) {
    GeometryData geometry;

    std::cout << "=== Loading Preprocessed Geometry ===" << std::endl;
    std::cout << "Loading geometry from: " << geometryFilePath << std::endl;

    std::ifstream file(geometryFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open geometry file: " << geometryFilePath << std::endl;
        return geometry;
    }

    std::string line;

    if (std::getline(file, line)) {
        if (line.rfind("vertices:", 0) == 0) {
            std::string vertices_data = line.substr(9);
            std::stringstream ss(vertices_data);
            float x, y, z;
            while (ss >> x >> y >> z) {
                geometry.vertices.push_back({x, y, z});
            }
        } else {
            std::cerr << "Error: Expected vertices line first" << std::endl;
            return GeometryData{};
        }
    }

    if (std::getline(file, line)) {
        if (line.rfind("indices:", 0) == 0) {
            std::string indices_data = line.substr(8);
            std::stringstream ss(indices_data);
            unsigned int i1, i2, i3;
            while (ss >> i1 >> i2 >> i3) {
                geometry.indices.push_back({i1, i2, i3});
            }
        } else {
            std::cerr << "Error: Expected indices line second" << std::endl;
            return GeometryData{};
        }
    }

    if (std::getline(file, line)) {
        if (line.rfind("triangleToObject:", 0) == 0) {
            std::string mapping_data = line.substr(17);
            std::stringstream ss(mapping_data);
            int objectId;
            while (ss >> objectId) {
                geometry.triangleToObject.push_back(objectId);
            }
        } else {
            std::cerr << "Error: Expected triangleToObject line third" << std::endl;
            return GeometryData{};
        }
    }

    if (std::getline(file, line)) {
        if (line.rfind("total_triangles:", 0) == 0) {
            std::string total_data = line.substr(16);
            std::stringstream ss(total_data);
            ss >> geometry.totalTriangles;
        } else {
            std::cerr << "Error: Expected total_triangles line fourth" << std::endl;
            return GeometryData{};
        }
    }

    // Optional Grid Data (appended by updated preprocessor)
    // We check the next line. If EOF, it's fine.
    if (std::getline(file, line)) {
        if (line.rfind("grid_info:", 0) == 0) {
            std::string grid_data = line.substr(10);
            std::stringstream ss(grid_data);
            float minx, miny, minz, maxx, maxy, maxz;
            unsigned int resx, resy, resz;
            ss >> minx >> miny >> minz >> maxx >> maxy >> maxz >> resx >> resy >> resz;
            
            geometry.grid.minBound = {minx, miny, minz};
            geometry.grid.maxBound = {maxx, maxy, maxz};
            geometry.grid.resolution = {resx, resy, resz};
            geometry.grid.hasGrid = true;
            
             if (std::getline(file, line)) {
                 if (line.rfind("grid_cells:", 0) == 0) {
                     std::string cell_data = line.substr(11);
                     std::stringstream ss2(cell_data);
                     size_t numCells = (size_t)resx * resy * resz;
                     geometry.grid.cells.reserve(numCells);
                     
                     uint32_t cc, tc;
                     float avgSize, volRatio;
                     while (ss2 >> cc >> tc >> avgSize >> volRatio) {
                         geometry.grid.cells.push_back({cc, tc, avgSize, volRatio});
                     }
                     
                     if (geometry.grid.cells.size() != numCells) {
                          std::cerr << "Warning: Grid cell count mismatch. Expected " << numCells << ", got " << geometry.grid.cells.size() << std::endl;
                     }
                 }
             }
        }
    }

    file.close();

    std::cout << "Loaded preprocessed geometry:" << std::endl;
    std::cout << "  Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "  Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "  Triangle-to-object mappings: " << geometry.triangleToObject.size() << std::endl;
    if (geometry.grid.hasGrid) {
        std::cout << "  Grid Statistics Loaded (" << geometry.grid.resolution.x << "x" << geometry.grid.resolution.y << "x" << geometry.grid.resolution.z << ")" << std::endl;
    }

    // Vectors are now pinned memory (via PinnedAllocator)
    // No need to copy to separate buffers or clear vectors.
    std::cout << "  Geometry loaded directly into pinned memory vectors" << std::endl;
    std::cout << "  Pinned buffers ready for GPU transfer (zero-copy)" << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
