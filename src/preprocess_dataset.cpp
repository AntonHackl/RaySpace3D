// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include "dataset_loader.h"
#include "timer.h"

void writeGeometryDataToFile(const GeometryData& geometry, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << std::fixed << std::setprecision(6);
    
    // Write vertices on the first line
    file << "vertices: ";
    for (size_t i = 0; i < geometry.vertices.size(); ++i) {
        file << geometry.vertices[i].x << " " 
             << geometry.vertices[i].y << " " 
             << geometry.vertices[i].z;
        if (i < geometry.vertices.size() - 1) file << " ";
    }
    file << "\n";
    
    // Write indices on the second line
    file << "indices: ";
    for (size_t i = 0; i < geometry.indices.size(); ++i) {
        file << geometry.indices[i].x << " " 
             << geometry.indices[i].y << " " 
             << geometry.indices[i].z;
        if (i < geometry.indices.size() - 1) file << " ";
    }
    file << "\n";
    
    // Write triangle to polygon mapping on the third line
    file << "triangleToPolygon: ";
    for (size_t i = 0; i < geometry.triangleToPolygon.size(); ++i) {
        file << geometry.triangleToPolygon[i];
        if (i < geometry.triangleToPolygon.size() - 1) file << " ";
    }
    file << "\n";
    
    // Write total triangles on the fourth line
    file << "total_triangles: " << geometry.totalTriangles << "\n";
    
    file.close();
    
    std::cout << "Geometry data written to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::string datasetPath = "";
    std::string outputGeometryPath = "geometry_data.txt";
    std::string outputTimingPath = "preprocessing_timing.json";
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                datasetPath = argv[++i];
            }
            else if (arg == "--output-geometry" && i + 1 < argc) {
                outputGeometryPath = argv[++i];
            }
            else if (arg == "--output-timing" && i + 1 < argc) {
                outputTimingPath = argv[++i];
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--dataset <path_to_wkt_file>] [--output-geometry <geometry_output_file>] [--output-timing <timing_output_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --dataset <path>           Path to WKT dataset file to triangulate" << std::endl;
                std::cout << "  --output-geometry <path>   Path to text file for geometry data output (default: geometry_data.txt)" << std::endl;
                std::cout << "  --output-timing <path>     Path to JSON file for preprocessing timing output (default: preprocessing_timing.json)" << std::endl;
                std::cout << "  --help, -h                 Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    if (datasetPath.empty()) {
        std::cerr << "Error: Dataset path is required. Use --dataset <path_to_wkt_file>" << std::endl;
        return 1;
    }
    
    std::cout << "Dataset Preprocessing Tool" << std::endl;
    std::cout << "Input dataset: " << datasetPath << std::endl;
    std::cout << "Output geometry: " << outputGeometryPath << std::endl;
    std::cout << "Output timing: " << outputTimingPath << std::endl;
    
    PerformanceTimer timer;
    // Phase 1: Load polygons only
    timer.start("Loading Polygons");
    auto polygons = loadPolygonsWithHoles(datasetPath);
    if (polygons.empty()) {
        std::cerr << "Error: Failed to load polygons from dataset." << std::endl;
        return 1;
    }

    // Phase 2: Triangulate polygons
    timer.next("Triangulating Polygons");
    GeometryData geometry = triangulatePolygons(polygons);
    if (geometry.vertices.empty()) {
        std::cerr << "Error: Failed to triangulate polygons from dataset." << std::endl;
        return 1;
    }

    // Phase 3: Write geometry data
    timer.next("Writing Geometry Data");
    
    // Write geometry data to file
    writeGeometryDataToFile(geometry, outputGeometryPath);
    
    timer.finish(outputTimingPath);
    
    std::cout << "\n=== Preprocessing Complete ===" << std::endl;
    std::cout << "Processed dataset: " << datasetPath << std::endl;
    std::cout << "Total polygons: " << geometry.totalTriangles << std::endl;
    std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "Geometry data saved to: " << outputGeometryPath << std::endl;
    std::cout << "Timing data saved to: " << outputTimingPath << std::endl;
    
    return 0;
}
