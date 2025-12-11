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

    file.close();

    std::cout << "Loaded preprocessed geometry:" << std::endl;
    std::cout << "  Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "  Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "  Triangle-to-object mappings: " << geometry.triangleToObject.size() << std::endl;
    
    // Allocate pinned memory and copy from std::vectors
    // Note: For geometry, we keep std::vector because parsing is complex
    // but we could optimize this further with direct pinned loading
    std::cout << "  Allocating pinned memory buffers..." << std::endl;
    geometry.pinnedBuffers.allocate(geometry.vertices.size(), 
                                    geometry.indices.size(), 
                                    geometry.triangleToObject.size());
    geometry.pinnedBuffers.copyFrom(geometry.vertices, 
                                    geometry.indices, 
                                    geometry.triangleToObject);
    
    // Free std::vectors immediately to reduce memory pressure
    geometry.vertices.clear();
    geometry.vertices.shrink_to_fit();
    geometry.indices.clear();
    geometry.indices.shrink_to_fit();
    geometry.triangleToObject.clear();
    geometry.triangleToObject.shrink_to_fit();
    
    std::cout << "  Pinned buffers ready for GPU transfer (std::vectors freed)" << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
