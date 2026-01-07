#include "ObjMeshDatasetLoader.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "Geometry.h"
#include "DatasetUtils.h"
#include <tiny_obj_loader.h>
#include <filesystem>

GeometryData ObjMeshDatasetLoader::load(const std::string& filePath) {
    GeometryData geometry;

    std::cout << "=== Mesh Dataset Loading ===" << std::endl;
    std::cout << "Loading mesh from file: " << filePath << std::endl;

    if (!std::filesystem::exists(filePath)) {
        std::cerr << "Error: File does not exist: " << filePath << std::endl;
        return geometry;
    }

    if (!std::filesystem::is_regular_file(filePath)) {
        std::cerr << "Error: Path is not a regular file: " << filePath << std::endl;
        return geometry;
    }

    if (std::filesystem::path(filePath).extension() != ".obj") {
        std::cerr << "Error: File is not a .obj file: " << filePath << std::endl;
        return geometry;
    }

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str());

    if (!err.empty()) {
        std::cerr << "Error loading .obj file: " << err << std::endl;
        return geometry;
    }

    if (!ret) {
        std::cerr << "Error: Failed to load .obj file: " << filePath << std::endl;
        return geometry;
    }

    if (shapes.empty()) {
        std::cerr << "Error: No shapes found in .obj file: " << filePath << std::endl;
        return geometry;
    }

    std::cout << "Found " << shapes.size() << " object(s) in file" << std::endl;

    int objectIndex = 0;
    int skippedShapes = 0;

    for (size_t shapeIdx = 0; shapeIdx < shapes.size(); ++shapeIdx) {
        const auto& shape = shapes[shapeIdx];
        if (shapeIdx % 100 == 0 || shapeIdx == shapes.size() - 1) {
            printProgressBar(shapeIdx + 1, shapes.size());
        }

        if (shape.mesh.num_face_vertices.empty()) {
            skippedShapes++;
            continue;
        }

        bool allTriangular = true;
        std::unordered_set<unsigned int> used_vertex_indices;
        size_t index_offset = 0;
        
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                allTriangular = false;
                break;
            }
            used_vertex_indices.insert(shape.mesh.indices[index_offset + 0].vertex_index);
            used_vertex_indices.insert(shape.mesh.indices[index_offset + 1].vertex_index);
            used_vertex_indices.insert(shape.mesh.indices[index_offset + 2].vertex_index);
            index_offset += fv;
        }

        if (!allTriangular || used_vertex_indices.empty()) {
            skippedShapes++;
            continue;
        }

        size_t vertexOffset = geometry.vertices.size();
        std::unordered_map<unsigned int, unsigned int> vertex_map;
        unsigned int local_idx = 0;

        for (unsigned int global_vidx : used_vertex_indices) {
            if (global_vidx * 3 + 2 < attrib.vertices.size()) {
                float3 vertex;
                vertex.x = attrib.vertices[3 * global_vidx + 0];
                vertex.y = attrib.vertices[3 * global_vidx + 1];
                vertex.z = attrib.vertices[3 * global_vidx + 2];
                geometry.vertices.push_back(vertex);
                vertex_map[global_vidx] = vertexOffset + local_idx++;
            }
        }

        index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            unsigned int v0_idx = shape.mesh.indices[index_offset + 0].vertex_index;
            unsigned int v1_idx = shape.mesh.indices[index_offset + 1].vertex_index;
            unsigned int v2_idx = shape.mesh.indices[index_offset + 2].vertex_index;
            
            auto it0 = vertex_map.find(v0_idx);
            auto it1 = vertex_map.find(v1_idx);
            auto it2 = vertex_map.find(v2_idx);
            
            if (it0 != vertex_map.end() && it1 != vertex_map.end() && it2 != vertex_map.end()) {
                geometry.indices.push_back({it0->second, it1->second, it2->second});
                geometry.triangleToObject.push_back(objectIndex);
            }
            index_offset += 3;
        }

        objectIndex++;
    }

    geometry.totalTriangles = geometry.indices.size();

    std::cout << "\n=== Mesh Loading Complete ===" << std::endl;
    std::cout << "Successfully loaded " << objectIndex << " object(s)" << std::endl;
    std::cout << "Skipped " << skippedShapes << " object(s)" << std::endl;
    std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "Total triangles: " << geometry.totalTriangles << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
