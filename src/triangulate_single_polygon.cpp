// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "triangulation.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>

int main(int argc, char* argv[]) {
    std::string datasetPath = "dtl_cnty.wkt";
    int targetPolygonIndex = 6; // Sixth polygon (0-indexed)
    
    if (argc > 1) {
        datasetPath = argv[1];
    }
    if (argc > 2) {
        targetPolygonIndex = std::atoi(argv[2]);
    }
    
    std::cout << "Reading polygons from: " << datasetPath << std::endl;
    std::cout << "Target polygon index: " << targetPolygonIndex << " (polygon #" << (targetPolygonIndex + 1) << ")" << std::endl;
    
    // Read all polygons from the file
    std::vector<PolygonWithHoles> allPolygons = readPolygonVerticesFromFile(datasetPath);
    
    if (allPolygons.empty()) {
        std::cerr << "No polygons found in the file!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << allPolygons.size() << " polygons" << std::endl;
    
    if (targetPolygonIndex >= allPolygons.size()) {
        std::cerr << "Requested polygon index " << targetPolygonIndex << " is out of range (0-" << (allPolygons.size() - 1) << ")" << std::endl;
        return 1;
    }
    
    // Get the target polygon
    const PolygonWithHoles& targetPolygon = allPolygons[targetPolygonIndex];
    
    std::cout << "Processing polygon #" << (targetPolygonIndex + 1) << ":" << std::endl;
    std::cout << "  Exterior vertices: " << targetPolygon.outer.size() << std::endl;
    std::cout << "  Interior holes: " << targetPolygon.holes.size() << std::endl;
    
    // Triangulate the polygon
    std::cout << "Triangulating polygon..." << std::endl;
    auto result = triangulatePolygon(targetPolygon);
    std::vector<Triangle> triangles = result.first;
    int method_used = result.second;
    
    std::cout << "Triangulation complete:" << std::endl;
    std::cout << "  Triangles: " << triangles.size() << std::endl;
    std::cout << "  Method used: " << method_used;
    if (method_used == 0) std::cout << " (CGAL Standard)";
    else if (method_used == 1) std::cout << " (CGAL Fallback)";
    else if (method_used == 2) std::cout << " (Unused)";
    else if (method_used == 3) std::cout << " (Failed)";
    std::cout << std::endl;
    
    // Save results to a simple text format for Python visualization
    std::ofstream outputFile("single_polygon_triangulation.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file for writing!" << std::endl;
        return 1;
    }
    
    // Set precision for floating point output
    outputFile << std::fixed << std::setprecision(6);
    
    // Write header
    outputFile << "# Single Polygon Triangulation Data" << std::endl;
    outputFile << "# Format: polygon_index polygon_number" << std::endl;
    outputFile << targetPolygonIndex << " " << (targetPolygonIndex + 1) << std::endl;
    
    // Write original polygon data
    outputFile << "# Original polygon exterior vertices" << std::endl;
    outputFile << "# Format: num_vertices" << std::endl;
    outputFile << targetPolygon.outer.size() << std::endl;
    outputFile << "# Format: x y" << std::endl;
    for (const auto& vertex : targetPolygon.outer) {
        outputFile << vertex.x << " " << vertex.y << std::endl;
    }
    
    // Write holes
    outputFile << "# Original polygon holes" << std::endl;
    outputFile << "# Format: num_holes" << std::endl;
    outputFile << targetPolygon.holes.size() << std::endl;
    
    for (const auto& hole : targetPolygon.holes) {
        outputFile << "# Hole with " << hole.size() << " vertices" << std::endl;
        outputFile << hole.size() << std::endl;
        for (const auto& vertex : hole) {
            outputFile << vertex.x << " " << vertex.y << std::endl;
        }
    }
    
    // Write triangulation data
    // Collect all unique vertices from the actual triangles
    std::vector<Point2D> allTriangulationVertices;
    std::map<std::pair<float, float>, int> vertexMap;
    
    // Extract all unique vertices from triangles
    for (const auto& triangle : triangles) {
        for (int i = 0; i < 3; ++i) {
            std::pair<float, float> vertexKey = {triangle.vertices[i].x, triangle.vertices[i].y};
            if (vertexMap.find(vertexKey) == vertexMap.end()) {
                vertexMap[vertexKey] = allTriangulationVertices.size();
                allTriangulationVertices.push_back(triangle.vertices[i]);
            }
        }
    }
    
    outputFile << "# Triangulation vertices (all vertices used in triangulation)" << std::endl;
    outputFile << "# Format: num_vertices" << std::endl;
    outputFile << allTriangulationVertices.size() << std::endl;
    outputFile << "# Format: x y" << std::endl;
    for (const auto& vertex : allTriangulationVertices) {
        outputFile << vertex.x << " " << vertex.y << std::endl;
    }
    
    // Write triangles
    outputFile << "# Triangulation triangles" << std::endl;
    outputFile << "# Format: num_triangles" << std::endl;
    outputFile << triangles.size() << std::endl;
    outputFile << "# Format: v1_x v1_y v2_x v2_y v3_x v3_y" << std::endl;
    for (const auto& triangle : triangles) {
        outputFile << triangle.vertices[0].x << " " << triangle.vertices[0].y << " "
                   << triangle.vertices[1].x << " " << triangle.vertices[1].y << " "
                   << triangle.vertices[2].x << " " << triangle.vertices[2].y << std::endl;
    }
    
    // Write boundary segments (constraints)
    outputFile << "# Boundary segments (constraints)" << std::endl;
    
    // Map original polygon vertices to triangulation vertices
    std::vector<int> originalVertexIndices;
    
    // Find indices of original exterior vertices in triangulation vertices
    for (const auto& originalVertex : targetPolygon.outer) {
        std::pair<float, float> vertexKey = {originalVertex.x, originalVertex.y};
        auto it = vertexMap.find(vertexKey);
        if (it != vertexMap.end()) {
            originalVertexIndices.push_back(it->second);
        }
    }
    
    // Find indices of hole vertices in triangulation vertices
    std::vector<std::vector<int>> holeVertexIndices;
    for (const auto& hole : targetPolygon.holes) {
        std::vector<int> holeIndices;
        for (const auto& holeVertex : hole) {
            std::pair<float, float> vertexKey = {holeVertex.x, holeVertex.y};
            auto it = vertexMap.find(vertexKey);
            if (it != vertexMap.end()) {
                holeIndices.push_back(it->second);
            }
        }
        if (!holeIndices.empty()) {
            holeVertexIndices.push_back(holeIndices);
        }
    }
    
    // Count total segments that can be mapped
    size_t totalSegments = 0;
    if (originalVertexIndices.size() == targetPolygon.outer.size()) {
        totalSegments += targetPolygon.outer.size();
    }
    for (size_t h = 0; h < holeVertexIndices.size(); ++h) {
        if (holeVertexIndices[h].size() == targetPolygon.holes[h].size()) {
            totalSegments += targetPolygon.holes[h].size();
        }
    }
    
    outputFile << "# Format: num_segments" << std::endl;
    outputFile << totalSegments << std::endl;
    outputFile << "# Format: v1 v2" << std::endl;
    
    // Exterior boundary segments
    if (originalVertexIndices.size() == targetPolygon.outer.size()) {
        for (size_t i = 0; i < originalVertexIndices.size(); ++i) {
            int nextIndex = (i + 1) % originalVertexIndices.size();
            outputFile << originalVertexIndices[i] << " " << originalVertexIndices[nextIndex] << std::endl;
        }
    }
    
    // Hole boundary segments
    for (size_t h = 0; h < holeVertexIndices.size(); ++h) {
        if (holeVertexIndices[h].size() == targetPolygon.holes[h].size()) {
            for (size_t i = 0; i < holeVertexIndices[h].size(); ++i) {
                int nextIndex = (i + 1) % holeVertexIndices[h].size();
                outputFile << holeVertexIndices[h][i] << " " << holeVertexIndices[h][nextIndex] << std::endl;
            }
        }
    }
    
    outputFile.close();
    
    std::cout << "Results saved to: single_polygon_triangulation.txt" << std::endl;
    std::cout << "You can now run the Python visualization script to view the results." << std::endl;
    
    return 0;
} 