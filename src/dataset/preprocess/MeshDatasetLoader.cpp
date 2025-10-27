#include "MeshDatasetLoader.h"

#include <iostream>

#include "../common/Geometry.h"
#include "../common/DatasetUtils.h"
#include <tiny_obj_loader.h>
#include <filesystem>

GeometryData MeshDatasetLoader::load(const std::string& directoryPath) {
    GeometryData geometry;

    std::cout << "=== Mesh Dataset Loading ===" << std::endl;
    std::cout << "Loading meshes from directory: " << directoryPath << std::endl;

    if (!std::filesystem::exists(directoryPath)) {
        std::cerr << "Error: Directory does not exist: " << directoryPath << std::endl;
        return geometry;
    }

    if (!std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Error: Path is not a directory: " << directoryPath << std::endl;
        return geometry;
    }

    std::vector<std::filesystem::path> objFiles;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {
            objFiles.push_back(entry.path());
        }
    }

    if (objFiles.empty()) {
        std::cerr << "Error: No .obj files found in directory: " << directoryPath << std::endl;
        return geometry;
    }

    std::cout << "Found " << objFiles.size() << " .obj files" << std::endl;

    int objectIndex = 0;
    int skippedFiles = 0;

    for (const auto& objFile : objFiles) {
        if (objectIndex >= 50) break; // Limit to first 50 objects
        printProgressBar(objectIndex + 1, objFiles.size());

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objFile.string().c_str());

        if (!err.empty()) { skippedFiles++; continue; }
        if (!ret) { skippedFiles++; continue; }

        bool allTriangulated = true;
        for (const auto& shape : shapes) {
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                if (shape.mesh.num_face_vertices[f] != 3) { allTriangulated = false; break; }
            }
            if (!allTriangulated) break;
        }
        if (!allTriangulated) { skippedFiles++; continue; }

        size_t vertexOffset = geometry.vertices.size();
        for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
            float3 vertex{attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]};
            geometry.vertices.push_back(vertex);
        }

        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv == 3) {
                    uint3 triangle{
                        static_cast<unsigned int>(vertexOffset + shape.mesh.indices[index_offset + 0].vertex_index),
                        static_cast<unsigned int>(vertexOffset + shape.mesh.indices[index_offset + 1].vertex_index),
                        static_cast<unsigned int>(vertexOffset + shape.mesh.indices[index_offset + 2].vertex_index)
                    };
                    geometry.indices.push_back(triangle);
                    geometry.triangleToObject.push_back(objectIndex);
                }
                index_offset += fv;
            }
        }
        objectIndex++;
    }

    geometry.totalTriangles = geometry.indices.size();

    std::cout << "\nComputing vertex normals for mesh triangles..." << std::endl;
    geometry.normals.resize(geometry.vertices.size(), {0.0f, 0.0f, 0.0f});

    for (const auto& triangle : geometry.indices) {
        const float3& v0 = geometry.vertices[triangle.x];
        const float3& v1 = geometry.vertices[triangle.y];
        const float3& v2 = geometry.vertices[triangle.z];

        float3 edge1{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        float3 edge2{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        float3 face_normal{
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x
        };
        geometry.normals[triangle.x].x += face_normal.x;
        geometry.normals[triangle.x].y += face_normal.y;
        geometry.normals[triangle.x].z += face_normal.z;
        geometry.normals[triangle.y].x += face_normal.x;
        geometry.normals[triangle.y].y += face_normal.y;
        geometry.normals[triangle.y].z += face_normal.z;
        geometry.normals[triangle.z].x += face_normal.x;
        geometry.normals[triangle.z].y += face_normal.y;
        geometry.normals[triangle.z].z += face_normal.z;
    }

    for (auto& normal : geometry.normals) {
        float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 0.0f) { normal.x /= length; normal.y /= length; normal.z /= length; }
    }

    std::cout << "Vertex normals computed by averaging face normals." << std::endl;
    std::cout << "\n=== Mesh Loading Complete ===" << std::endl;
    std::cout << "Successfully loaded " << objectIndex << " .obj files" << std::endl;
    std::cout << "Skipped " << skippedFiles << " files" << std::endl;
    std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "Total triangles: " << geometry.totalTriangles << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
