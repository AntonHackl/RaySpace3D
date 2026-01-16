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

    // Optional: Euler Histogram Data (Read until EOF)
    while (std::getline(file, line)) {
        if (line.rfind("euler_grid_dims:", 0) == 0) {
            std::stringstream ss(line.substr(16));
            ss >> geometry.eulerHistogram.nx >> geometry.eulerHistogram.ny >> geometry.eulerHistogram.nz;
        }
        else if (line.rfind("euler_bbox:", 0) == 0) {
            std::stringstream ss(line.substr(11));
            ss >> geometry.eulerHistogram.minBound.x >> geometry.eulerHistogram.minBound.y >> geometry.eulerHistogram.minBound.z
                >> geometry.eulerHistogram.maxBound.x >> geometry.eulerHistogram.maxBound.y >> geometry.eulerHistogram.maxBound.z;

            // Recompute cell size
            if (geometry.eulerHistogram.nx > 0) {
                float3& minB = geometry.eulerHistogram.minBound;
                float3& maxB = geometry.eulerHistogram.maxBound;
                geometry.eulerHistogram.cellSize = {
                    (maxB.x - minB.x) / geometry.eulerHistogram.nx,
                    (maxB.y - minB.y) / geometry.eulerHistogram.ny,
                    (maxB.z - minB.z) / geometry.eulerHistogram.nz
                };
            }
        }
        else if (line.rfind("euler_data_v:", 0) == 0) {
            std::stringstream ss(line.substr(13));
            int val;
            while (ss >> val) geometry.eulerHistogram.v_counts.push_back(val);
        }
        else if (line.rfind("euler_data_e:", 0) == 0) {
            std::stringstream ss(line.substr(13));
            int val;
            while (ss >> val) geometry.eulerHistogram.e_counts.push_back(val);
        }
        else if (line.rfind("euler_data_f:", 0) == 0) {
            std::stringstream ss(line.substr(13));
            int val;
            while (ss >> val) geometry.eulerHistogram.f_counts.push_back(val);
        }
        else if (line.rfind("euler_data_o:", 0) == 0) {
            std::stringstream ss(line.substr(13));
            int val;
            while (ss >> val) geometry.eulerHistogram.object_counts.push_back(val);
        }
    }

    file.close();

    std::cout << "Loaded preprocessed geometry:" << std::endl;
    std::cout << "  Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "  Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "  Triangle-to-object mappings: " << geometry.triangleToObject.size() << std::endl;
    if (geometry.eulerHistogram.nx > 0) {
        std::cout << "  Euler Histogram loaded: " << geometry.eulerHistogram.nx << "x"
            << geometry.eulerHistogram.ny << "x" << geometry.eulerHistogram.nz << " grid" << std::endl;
    }

    // Vectors are now pinned memory (via PinnedAllocator)
    // No need to copy to separate buffers or clear vectors.
    std::cout << "  Geometry loaded directly into pinned memory vectors" << std::endl;
    std::cout << "  Pinned buffers ready for GPU transfer (zero-copy)" << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
