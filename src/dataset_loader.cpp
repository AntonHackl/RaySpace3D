#include "dataset_loader.h"
#include "triangulation.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <map>
#include <array>

using namespace std;

// Simple cross-platform progress bar
void printProgressBar(size_t current, size_t total, int barWidth = 50) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) 
              << (progress * 100.0) << "% (" << current << "/" << total << ")";
    std::cout.flush();
    
    if (current == total) {
        std::cout << std::endl;
    }
}

// Load polygons only
std::vector<PolygonWithHoles> loadPolygonsWithHoles(const std::string& datasetPath) {
    std::vector<PolygonWithHoles> polygons;
    if (datasetPath.empty()) {
        std::cerr << "Error: Dataset path is empty." << std::endl;
        return polygons;
    }
    std::cout << "=== Dataset Loading ===" << std::endl;
    std::cout << "Loading polygons from: " << datasetPath << std::endl;
    polygons = readPolygonVerticesFromFile(datasetPath);
    if (polygons.empty()) {
        std::cerr << "Error: No valid polygons found in dataset file." << std::endl;
    } else {
        std::cout << "Found " << polygons.size() << " polygons in dataset" << std::endl;
    }
    std::cout << "=======================\n" << std::endl;
    return polygons;
}

// Triangulate loaded polygons
GeometryData triangulatePolygons(const std::vector<PolygonWithHoles>& polygons) {
    GeometryData geometry;
    if (polygons.empty()) {
        std::cerr << "Error: No polygons provided for triangulation." << std::endl;
        return geometry;
    }
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
            geometry.triangleToPolygon.push_back(static_cast<int>(original_poly_idx));
        }
    }
    std::cout << "Dataset converted to " << geometry.vertices.size() << " vertices and " << geometry.indices.size() << " triangles" << std::endl;
    std::cout << "Using dataset triangles for raytracing acceleration structure" << std::endl;
    std::cout << "Geometry loaded: " << geometry.vertices.size() << " vertices, " << geometry.indices.size() << " triangles" << std::endl;
    return geometry;
}

// Backwards compatible wrapper
GeometryData loadDatasetGeometry(const std::string& datasetPath) {
    if (datasetPath.empty()) {
        GeometryData geometry; // fallback single triangle
        geometry.vertices = { {0.0f,0.0f,0.0f},{0.5f,1.0f,0.0f},{1.0f,0.0f,0.0f} };
        geometry.indices = { {0,1,2} };
        geometry.triangleToPolygon = {0};
        geometry.totalTriangles = 1;
        std::cout << "Using fallback single triangle (no dataset provided)" << std::endl;
        return geometry;
    }
    auto polys = loadPolygonsWithHoles(datasetPath);
    if (polys.empty()) return GeometryData{};
    return triangulatePolygons(polys);
}

// Point dataset loader (unchanged from original version)
PointData loadPointDataset(const std::string& pointDatasetPath) {
    PointData pointData;
    if (pointDatasetPath.empty()) {
        std::cout << "No point dataset provided, using default test points" << std::endl;
        pointData.positions = { {0.0f,0.0f,-1.0f},{0.5f,0.5f,-1.0f},{1.0f,1.0f,-1.0f} };
        pointData.numPoints = 3; return pointData; }
    std::cout << "=== Loading Point Dataset ===" << std::endl;
    std::cout << "Loading points from: " << pointDatasetPath << std::endl;
    std::ifstream file(pointDatasetPath);
    if (!file.is_open()) { std::cerr << "Error: Could not open point dataset file: " << pointDatasetPath << std::endl; return pointData; }
    std::string line; int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0]=='#') continue;
        if (line.find("POINT") != std::string::npos) {
            try {
                size_t start = line.find('('); size_t end = line.find(')');
                if (start!=std::string::npos && end!=std::string::npos) {
                    std::string coords = line.substr(start+1, end-start-1);
                    size_t spacePos = coords.find(' ');
                    if (spacePos != std::string::npos) {
                        float x = std::stof(coords.substr(0, spacePos));
                        float y = std::stof(coords.substr(spacePos+1));
                        pointData.positions.push_back({x,y,-1.0f});
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse point on line " << lineNum << ": " << e.what() << std::endl; continue;
            }
        }
    }
    pointData.numPoints = pointData.positions.size();
    std::cout << "Loaded " << pointData.numPoints << " points from dataset" << std::endl;
    std::cout << "=============================\n" << std::endl;
    return pointData;
}

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
    
    // Read vertices line
    if (std::getline(file, line)) {
        if (line.substr(0, 9) == "vertices:") {
            std::string vertices_data = line.substr(9); // Remove "vertices:" prefix
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
    
    // Read indices line
    if (std::getline(file, line)) {
        if (line.substr(0, 8) == "indices:") {
            std::string indices_data = line.substr(8); // Remove "indices:" prefix
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
    
    // Read triangleToPolygon line
    if (std::getline(file, line)) {
        if (line.substr(0, 18) == "triangleToPolygon:") {
            std::string mapping_data = line.substr(18); // Remove "triangleToPolygon:" prefix
            std::stringstream ss(mapping_data);
            int polygonId;
            while (ss >> polygonId) {
                geometry.triangleToPolygon.push_back(polygonId);
            }
        } else {
            std::cerr << "Error: Expected triangleToPolygon line third" << std::endl;
            return GeometryData{};
        }
    }
    
    // Read total_triangles line
    if (std::getline(file, line)) {
        if (line.substr(0, 16) == "total_triangles:") {
            std::string total_data = line.substr(16); // Remove "total_triangles:" prefix
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
    std::cout << "  Triangle-to-polygon mappings: " << geometry.triangleToPolygon.size() << std::endl;
    std::cout << "=============================\n" << std::endl;
    
    return geometry;
} 