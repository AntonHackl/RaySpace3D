#include "PolygonDatasetLoader.h"

#include <iostream>
#include <vector>

#include "../../triangulation.h"
#include "../common/Geometry.h"
#include "../common/DatasetUtils.h"

GeometryData PolygonDatasetLoader::load(const std::string& wktFilePath) {
    GeometryData geometry;
    if (wktFilePath.empty()) {
        std::cerr << "Error: WKT dataset path is empty." << std::endl;
        return geometry;
    }

    std::cout << "=== Dataset Loading ===" << std::endl;
    std::cout << "Loading polygons from: " << wktFilePath << std::endl;
    auto polygons = readPolygonVerticesFromFile(wktFilePath);
    if (polygons.empty()) {
        std::cerr << "Error: No valid polygons found in dataset file." << std::endl;
        return geometry;
    } else {
        std::cout << "Found " << polygons.size() << " polygons in dataset" << std::endl;
    }
    std::cout << "=======================\n" << std::endl;

    // Triangulate
    if (polygons.empty()) return geometry;
    std::cout << "=== Dataset Triangulation ===" << std::endl;
    std::cout << "Triangulating polygons..." << std::endl;

    std::vector<std::vector<Triangle>> triangulated_polygons;
    std::vector<size_t> valid_polygon_indices;
    TriangulationStats stats;

    for (size_t poly_idx = 0; poly_idx < polygons.size(); ++poly_idx) {
        if (poly_idx % 100 == 0 || poly_idx == polygons.size() - 1)  {
            printProgressBar(poly_idx + 1, polygons.size());
        }
        const auto& poly = polygons[poly_idx];
        try {
            auto result = triangulatePolygon(poly);
            auto triangulated = result.first;
            int method_used = result.second;
            if (method_used == 0) stats.cgal_success++;
            else if (method_used == 1) stats.cgal_repaired++;
            else if (method_used == 2) stats.cgal_decomposed++;
            else if (method_used == 3) stats.failed_method++;
            if (triangulated.empty()) continue;
            bool valid_triangulation = true;
            for (const auto& tri : triangulated) {
                if ((tri.vertices[0].x == tri.vertices[1].x && tri.vertices[0].y == tri.vertices[1].y) ||
                    (tri.vertices[1].x == tri.vertices[2].x && tri.vertices[1].y == tri.vertices[2].y) ||
                    (tri.vertices[2].x == tri.vertices[0].x && tri.vertices[2].y == tri.vertices[0].y)) { valid_triangulation = false; break; }
            }
            if (!valid_triangulation) continue;
            triangulated_polygons.push_back(triangulated);
            valid_polygon_indices.push_back(poly_idx);
        } catch (const std::exception&) { continue; }
    }
    std::cout << std::endl;
    stats.print();
    if (triangulated_polygons.empty()) {
        std::cerr << "Error: No polygons could be successfully triangulated." << std::endl;
        return geometry;
    }
    std::cout << "Successfully triangulated " << triangulated_polygons.size() << " out of " << polygons.size() << " polygons" << std::endl;
    geometry.totalTriangles = countTriangles(triangulated_polygons);
    std::cout << "Total number of triangles: " << geometry.totalTriangles << std::endl;
    std::cout << "=============================\n" << std::endl;
    std::cout << "Converting dataset triangles to OptiX format..." << std::endl;
    for (size_t valid_idx = 0; valid_idx < triangulated_polygons.size(); ++valid_idx) {
        const auto& triangles = triangulated_polygons[valid_idx];
        const size_t original_poly_idx = valid_polygon_indices[valid_idx];
        std::map<std::pair<float,float>, size_t> vertex_map;
        std::vector<Point2D> polygon_vertices;
        for (const auto& tri : triangles) {
            for (int i=0;i<3;++i) {
                std::pair<float,float> key{tri.vertices[i].x, tri.vertices[i].y};
                if (vertex_map.find(key)==vertex_map.end()) { vertex_map[key]=polygon_vertices.size(); polygon_vertices.push_back(tri.vertices[i]); }
            }
        }
        size_t vertex_offset = geometry.vertices.size();
        for (const auto& v : polygon_vertices) geometry.vertices.push_back({v.x, v.y, 0.0f});
        for (const auto& tri : triangles) {
            std::array<unsigned int,3> idx;
            for (int i=0;i<3;++i) { std::pair<float,float> key{tri.vertices[i].x, tri.vertices[i].y}; idx[i] = static_cast<unsigned int>(vertex_offset + vertex_map[key]); }
            geometry.indices.push_back({idx[0], idx[1], idx[2]});
            geometry.triangleToObject.push_back(static_cast<int>(original_poly_idx));
        }
    }
    std::cout << "Dataset converted to " << geometry.vertices.size() << " vertices and " << geometry.indices.size() << " triangles" << std::endl;

    std::cout << "Computing vertex normals for z=0 plane polygons..." << std::endl;
    geometry.normals.resize(geometry.vertices.size(), {0.0f, 0.0f, 0.0f});

    for (const auto& triangle : geometry.indices) {
        const float3& v0 = geometry.vertices[triangle.x];
        const float3& v1 = geometry.vertices[triangle.y];
        const float3& v2 = geometry.vertices[triangle.z];

        float edge1_x = v1.x - v0.x;
        float edge1_y = v1.y - v0.y;
        float edge2_x = v2.x - v0.x;
        float edge2_y = v2.y - v0.y;

        float cross_z = edge1_x * edge2_y - edge1_y * edge2_x;
        float normal_z = (cross_z > 0.0f) ? 1.0f : -1.0f;

        geometry.normals[triangle.x].z += normal_z;
        geometry.normals[triangle.y].z += normal_z;
        geometry.normals[triangle.z].z += normal_z;
    }

    for (auto& normal : geometry.normals) {
        normal.z = (normal.z > 0.0f) ? 1.0f : -1.0f;
    }
    std::cout << "Vertex normals computed (all pointing in +z or -z direction)." << std::endl;

    std::cout << "Using dataset triangles for raytracing acceleration structure" << std::endl;
    std::cout << "Geometry loaded: " << geometry.vertices.size() << " vertices, " << geometry.indices.size() << " triangles" << std::endl;
    return geometry;
}
