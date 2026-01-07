// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../geometry/BoundingBox.h"
#include "../raytracing/FilterRefine.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "PointIO.h"
#include "../timer.h"
#include "../ptx_utils.h"

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");
    
    std::string geometryFilePath = "";
    std::string pointDatasetPath = "";
    std::string outputJsonPath = "performance_timing_filter_refine.json";
    std::string ptxPath = detectPTXPath();
    int numberOfRuns = 1;
    bool exportResults = true;
    int warmupRuns = 2;
    
    // Parse command line arguments
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--geometry" && i + 1 < argc) {
                geometryFilePath = argv[++i];
            }
            else if (arg == "--points" && i + 1 < argc) {
                pointDatasetPath = argv[++i];
            }
            else if (arg == "--output" && i + 1 < argc) {
                outputJsonPath = argv[++i];
            }
            else if (arg == "--runs" && i + 1 < argc) {
                numberOfRuns = std::atoi(argv[++i]);
            }
            else if (arg == "--warmup-runs" && i + 1 < argc) {
                warmupRuns = std::atoi(argv[++i]);
            }
            else if (arg == "--no-export") {
                exportResults = false;
            }
            else if (arg == "--ptx" && i + 1 < argc) {
                ptxPath = argv[++i];
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--geometry <path>] [--points <path>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --geometry <path>     Path to preprocessed geometry text file." << std::endl;
                std::cout << "  --points <path>       Path to WKT file containing POINT geometries." << std::endl;
                std::cout << "  --output <path>       Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>       Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --warmup-runs <num>   Number of warmup runs (default: 2)" << std::endl;
                std::cout << "  --no-export           Disable CSV export of results" << std::endl;
                std::cout << "  --ptx <ptx_file>      Path to compiled PTX file (default: ./raytracing.ptx)" << std::endl;
                std::cout << "  --help, -h            Show this help message" << std::endl;
                std::cout << "\nThis program implements a filter-refine approach:" << std::endl;
                std::cout << "  1. Filter: Test points against query geometry bounding box" << std::endl;
                std::cout << "  2. Refine: Perform exact raytracing for points inside bounding box" << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "OptiX Filter-Refine Ray Tracing" << std::endl;
    
    if (geometryFilePath.empty()) {
        std::cerr << "Error: Geometry file path is required. Use --geometry <path_to_geometry_file>" << std::endl;
        return 1;
    }
    
    if (pointDatasetPath.empty()) {
        std::cerr << "Error: Points file path is required. Use --points <path_to_point_file>" << std::endl;
        return 1;
    }
    
    // Load geometry
    std::cout << "Loading geometry from: " << geometryFilePath << std::endl;
    GeometryData geometry = loadGeometryFromFile(geometryFilePath);
    if (geometry.vertices.empty()) {
        std::cerr << "Error: Failed to load geometry from " << geometryFilePath << std::endl;
        return 1;
    }
    std::cout << "Geometry loaded: " << geometry.vertices.size() << " vertices, " 
              << geometry.indices.size() << " triangles" << std::endl;
    
    // Load points
    std::cout << "Loading points from: " << pointDatasetPath << std::endl;
    PointData pointData = loadPointDataset(pointDatasetPath);
    if (pointData.numPoints == 0) {
        std::cerr << "Error: Failed to load points from " << pointDatasetPath << std::endl;
        return 1;
    }
    std::cout << "Points loaded: " << pointData.numPoints << std::endl;
    
    timer.next("Application Creation");
    
    // Initialize OptiX
    OptixContext context;
    OptixPipelineManager pipeline(context, ptxPath);
    FilterRefinePipeline filterRefine(context, pipeline);
    
    // Compute bounding box
    BoundingBox queryBBox = BoundingBox::computeFromGeometry(geometry);
    std::cout << "\nQuery bounding box computed:" << std::endl;
    std::cout << "  Min: (" << queryBBox.min.x << ", " << queryBBox.min.y << ", " << queryBBox.min.z << ")" << std::endl;
    std::cout << "  Max: (" << queryBBox.max.x << ", " << queryBBox.max.y << ", " << queryBBox.max.z << ")" << std::endl;
    
    timer.next("Filter-Refine Execution");
    
    // Execute filter-refine pipeline
    FilterRefineResult result = filterRefine.execute(geometry, pointData);
    
    timer.next("Output");
    
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Total points:            " << result.totalPoints << std::endl;
    std::cout << "Filtered by bbox:        " << result.filteredOut << " (" 
              << (result.totalPoints > 0 ? (result.filteredOut * 100.0 / result.totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Candidates tested:       " << result.candidateCount << " (" 
              << (result.totalPoints > 0 ? (result.candidateCount * 100.0 / result.totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Points INSIDE polygons:  " << result.insideCount << " (" 
              << (result.totalPoints > 0 ? (result.insideCount * 100.0 / result.totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Points OUTSIDE polygons: " << result.outsideCount << " (" 
              << (result.totalPoints > 0 ? (result.outsideCount * 100.0 / result.totalPoints) : 0.0) << "%)" << std::endl;
    
    if (exportResults) {
        std::cout << "Exporting results (hits only)" << std::endl;
        std::ofstream csvFile("ray_results.csv");
        csvFile << "pointId,polygonId\n";
        for (const auto& hit : result.refineHits) {
            if (hit.ray_id >= 0 && static_cast<size_t>(hit.ray_id) < result.candidateIndices.size()) {
                int originalIdx = result.candidateIndices[hit.ray_id];
                csvFile << originalIdx << "," << hit.polygon_index << "\n";
            }
        }
        csvFile.close();
        std::cout << "Exported " << result.insideCount << " hit points" << std::endl;
    }
    
    timer.next("Cleanup");
    
    // Pinned vectors free themselves
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Filter-Refine processing complete!" << std::endl;
    if (numberOfRuns > 1) {
        std::cout << "Number of runs: " << numberOfRuns << std::endl;
    }

    return 0;
}

