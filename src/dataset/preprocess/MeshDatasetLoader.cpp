#include "MeshDatasetLoader.h"

#include <iostream>
#include <map>

#include "../common/Geometry.h"
#include "../common/DatasetUtils.h"
#include <tiny_obj_loader.h>
#include <filesystem>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Surface_mesh;

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
    int currentIndex = 0;

    for (const auto& objFile : objFiles) {
        if (objectIndex >= 50) break;
        printProgressBar(++currentIndex, objFiles.size());

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

        Surface_mesh mesh;
        std::vector<Surface_mesh::Vertex_index> vertex_indices;
        vertex_indices.reserve(attrib.vertices.size() / 3);
        for (size_t v = 0; v < attrib.vertices.size() / 3; ++v) {
            K::Point_3 p(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
            vertex_indices.push_back(mesh.add_vertex(p));
        }

        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) { skippedFiles++; mesh.clear(); break; }
                std::vector<Surface_mesh::Vertex_index> face_verts;
                face_verts.push_back(vertex_indices[shape.mesh.indices[index_offset + 0].vertex_index]);
                face_verts.push_back(vertex_indices[shape.mesh.indices[index_offset + 1].vertex_index]);
                face_verts.push_back(vertex_indices[shape.mesh.indices[index_offset + 2].vertex_index]);
                auto fdesc = mesh.add_face(face_verts);
                (void)fdesc;
                index_offset += fv;
            }
            if (mesh.number_of_faces() == 0) break;
        }

        mesh.collect_garbage();

        if (mesh.number_of_faces() == 0) {
            skippedFiles++;
            continue;
        }

        if (!(CGAL::is_closed(mesh) && CGAL::is_valid_polygon_mesh(mesh))) {
            skippedFiles++;
            continue;
        }
        if (CGAL::Polygon_mesh_processing::does_self_intersect(mesh)) {
            skippedFiles++;
            continue;
        }

        try {
            CGAL::Polygon_mesh_processing::orient_to_bound_a_volume(mesh);
        } catch (const CGAL::Precondition_exception& /*e*/) {
            skippedFiles++;
            continue;
        } catch (...) {
            skippedFiles++;
            continue;
        }

        mesh.collect_garbage();
        auto fnormals = mesh.add_property_map<Surface_mesh::Face_index, K::Vector_3>("f:normals", CGAL::NULL_VECTOR).first;
        CGAL::Polygon_mesh_processing::compute_face_normals(mesh, fnormals);

        size_t vertexOffset = geometry.vertices.size();
        for (auto v : mesh.vertices()) {
            const K::Point_3& p = mesh.point(v);
            geometry.vertices.push_back({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())});
        }

        std::map<Surface_mesh::Vertex_index, unsigned int> vertex_map;
        unsigned int idx = 0;
        for (auto v : mesh.vertices()) {
            vertex_map[v] = vertexOffset + idx++;
        }

        size_t startNormalIdx = geometry.normals.size();
        for (auto f : mesh.faces()) {
            std::vector<unsigned int> face_indices;
            for (auto v : vertices_around_face(mesh.halfedge(f), mesh)) {
                face_indices.push_back(vertex_map[v]);
            }
            if (face_indices.size() == 3) {
                geometry.indices.push_back({face_indices[0], face_indices[1], face_indices[2]});
                geometry.triangleToObject.push_back(objectIndex);
                
                const K::Vector_3& n = fnormals[f];
                float length = static_cast<float>(std::sqrt(n.squared_length()));
                if (length > 0.0f) {
                    geometry.normals.push_back({
                        static_cast<float>(n.x() / length),
                        static_cast<float>(n.y() / length),
                        static_cast<float>(n.z() / length)
                    });
                } else {
                    geometry.normals.push_back({0.0f, 0.0f, 1.0f});
                }
            }
        }

        objectIndex++;
    }

    geometry.totalTriangles = geometry.indices.size();

    std::cout << "\nFace normals computed (one per triangle)." << std::endl;
    std::cout << "\n=== Mesh Loading Complete ===" << std::endl;
    std::cout << "Successfully loaded " << objectIndex << " .obj files" << std::endl;
    std::cout << "Skipped " << skippedFiles << " files" << std::endl;
    std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "Total triangles: " << geometry.totalTriangles << std::endl;
    std::cout << "=============================\n" << std::endl;

    return geometry;
}
