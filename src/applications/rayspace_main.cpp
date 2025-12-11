// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "../geometry/GeometryUploader.h"
#include "../raytracing/RayLauncher.h"
#include "../raytracing/ResultProcessor.h"
#include "../dataset/common/Geometry.h"
#include "../dataset/runtime/GeometryIO.h"
#include "../dataset/runtime/PointIO.h"
#include "../timer.h"
#include "../ptx_utils.h"

// Helper function to split comma-separated file paths
static std::vector<std::string> splitPaths(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            result.push_back(item.substr(start, end - start + 1));
        } else if (start != std::string::npos) {
            result.push_back(item.substr(start));
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");
    
    std::string geometryFilePath = "";
    std::string pointDatasetPath = "";
    std::string outputJsonPath = "performance_timing.json";
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
            else if (arg == "--ptx" && i + 1 < argc) {
                ptxPath = argv[++i];
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
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--geometry <path(s)>] [--points <path(s)>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --geometry <path(s)>  Path(s) to preprocessed geometry text file(s). Use commas to separate multiple files." << std::endl;
                std::cout << "  --points <path(s)>    Path(s) to WKT file(s) containing POINT geometries. Use commas to separate multiple files." << std::endl;
                std::cout << "  --output <path>       Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>       Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --ptx <ptx_file>      Path to compiled PTX file (default: ./raytracing.ptx)" << std::endl;
                std::cout << "  --help, -h            Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "OptiX multiple rays example" << std::endl;

    // Parse file lists
    std::vector<std::string> geometryFiles = splitPaths(geometryFilePath);
    std::vector<std::string> pointFiles = splitPaths(pointDatasetPath);
    
    if (geometryFiles.empty()) {
        std::cerr << "Error: Geometry file path is required. Use --geometry <path_to_geometry_file>" << std::endl;
        return 1;
    }
    
    if (pointFiles.empty()) {
        std::cerr << "Error: Points file path is required. Use --points <path_to_point_file>" << std::endl;
        return 1;
    }
    
    // Validate list lengths
    size_t numTasks = std::max(geometryFiles.size(), pointFiles.size());
    if (geometryFiles.size() > 1 && pointFiles.size() > 1 && geometryFiles.size() != pointFiles.size()) {
        std::cerr << "Error: When both --geometry and --points are lists, they must have the same length." << std::endl;
        return 1;
    }
    
    std::cout << "Processing " << numTasks << " task(s)" << std::endl;
    
    timer.next("Application Creation");
    
    // Initialize OptiX
    OptixContext context;
    OptixPipelineManager pipeline(context, ptxPath);
    
    // Caching variables for geometry and points
    std::string cachedGeometryPath = "";
    GeometryData cachedGeometry;
    GeometryUploader geometryUploader;
    OptixAccelerationStructure* geometryAS = nullptr;
    
    std::string cachedPointsPath = "";
    PointData cachedPointData;
    RayLauncher* launcher = nullptr;
    
    ResultProcessor resultProcessor;
    
    // Process each task
    for (size_t taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
        std::string currentGeomPath = geometryFiles[geometryFiles.size() == 1 ? 0 : taskIdx];
        std::string currentPointsPath = pointFiles[pointFiles.size() == 1 ? 0 : taskIdx];
        
        std::cout << "\n=== Task " << (taskIdx + 1) << "/" << numTasks << " ===" << std::endl;
        std::cout << "Geometry: " << currentGeomPath << std::endl;
        std::cout << "Points: " << currentPointsPath << std::endl;
        
        // Load/cache geometry
        bool geometryChanged = (currentGeomPath != cachedGeometryPath);
        if (geometryChanged) {
            timer.next("Load Geometry");
            std::cout << "Loading new geometry..." << std::endl;
            
            cachedGeometry = loadGeometryFromFile(currentGeomPath);
            if (!cachedGeometry.pinnedBuffers.allocated || cachedGeometry.pinnedBuffers.vertices_size == 0) {
                std::cerr << "Error: Failed to load geometry from " << currentGeomPath << std::endl;
                continue;
            }
            cachedGeometryPath = currentGeomPath;
            
            // Upload geometry
            timer.next("Upload Geometry");
            geometryUploader.upload(cachedGeometry);
            
            // Build acceleration structure
            timer.next("Build Index");
            if (geometryAS) delete geometryAS;
            geometryAS = new OptixAccelerationStructure(context, geometryUploader);
            geometryAS->build();
            std::cout << "Acceleration structure built" << std::endl;
        } else {
            std::cout << "Using cached geometry" << std::endl;
        }
        
        // Load/cache points
        bool pointsChanged = (currentPointsPath != cachedPointsPath);
        if (pointsChanged) {
            timer.next("Load Points");
            std::cout << "Loading new points..." << std::endl;
            
            cachedPointData = loadPointDataset(currentPointsPath);
            if (cachedPointData.numPoints == 0 || !cachedPointData.pinnedBuffers.allocated) {
                std::cerr << "Error: Failed to load points from " << currentPointsPath << std::endl;
                continue;
            }
            cachedPointsPath = currentPointsPath;
            std::cout << "Points loaded: " << cachedPointData.numPoints << std::endl;
            
            // Upload points
            timer.next("Upload Points");
            if (launcher) delete launcher;
            launcher = new RayLauncher(pipeline, geometryUploader);
            launcher->uploadRays(cachedPointData.pinnedBuffers.positions_pinned, cachedPointData.numPoints);
            std::cout << "Points uploaded to GPU" << std::endl;
        } else {
            std::cout << "Using cached points" << std::endl;
        }
        
        const int numRays = static_cast<int>(cachedPointData.numPoints);
        
        // Warmup runs
        timer.next("Warmup");
        if (launcher && geometryAS) {
            launcher->runWarmup(*geometryAS, warmupRuns);
        }
        
        timer.next("Query");
        
        // Execute query
        if (numberOfRuns > 1) {
            std::cout << "\n=== Running " << numberOfRuns << " iterations for performance measurement ===" << std::endl;
            for (int run = 0; run < numberOfRuns; ++run) {
                std::cout << "Run " << (run + 1) << "/" << numberOfRuns << std::endl;
                if (launcher && geometryAS) {
                    launcher->launch(*geometryAS, numRays);
                }
            }
        } else {
            std::cout << "\n=== Tracing " << numRays << " rays in a single GPU launch ===" << std::endl;
            if (launcher && geometryAS) {
                launcher->launch(*geometryAS, numRays);
            }
        }
        
        timer.next("Download Results");
        
        // Process results
        std::vector<RayResult> hits;
        if (launcher) {
            hits = resultProcessor.compactAndDownload(launcher->getResultBuffer(), numRays);
        }
        
        int numHit = static_cast<int>(hits.size());
        size_t numMiss = numRays - numHit;
        
        timer.next("Output");
        std::cout << "\n=== Point-in-Polygon Summary ===" << std::endl;
        std::cout << "Total rays: " << numRays << std::endl;
        std::cout << "Points INSIDE polygons:  " << numHit << std::endl;
        std::cout << "Points OUTSIDE polygons: " << numMiss << std::endl;
        std::cout << "Inside ratio: " << (numRays > 0 ? (double)numHit / numRays * 100.0 : 0.0) << " %" << std::endl;
        
        if (exportResults) {
            std::cout << "Exporting results (hits only)" << std::endl;
            std::ofstream csvFile("ray_results.csv");
            csvFile << "pointId,polygonId\n";
            for (const auto& hit : hits) {
                csvFile << hit.ray_id << "," << hit.polygon_index << "\n";
            }
            csvFile.close();
        }
        
        // Free pinned memory buffers after task completion
        cachedGeometry.pinnedBuffers.free();
        cachedPointData.pinnedBuffers.free();
    }
    
    timer.next("Cleanup");
    
    // Cleanup
    if (launcher) delete launcher;
    if (geometryAS) delete geometryAS;
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Total tasks processed: " << numTasks << std::endl;
    
    return 0;
}

