// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_overlap.h"
#include "../cuda/selectivity_estimation.h"
#include "../cuda/mesh_overlap_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshOverlapLauncher.h"
#include "../timer.h"
#include "../ptx_utils.h"

// Structure to hold query results
// Structure to hold query results
struct QueryResults {
    MeshOverlapResult* d_merged_results;
    int numUnique;
};

// Execute the overlap query using hash table deduplication
QueryResults executeHashQuery(
    MeshOverlapLauncher& overlapLauncher,
    MeshOverlapLaunchParams& params1,
    MeshOverlapLaunchParams& params2,
    int mesh1NumTriangles,
    int mesh2NumTriangles,
    unsigned long long* d_hash_table,
    int hash_table_size,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    // Ensure params use hash table
    params1.use_hash_table = true;
    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    
    params2.use_hash_table = true;
    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;

    // Launch kernels (Single pass; pass param is ignored by kernel when use_hash_table is true)
    overlapLauncher.launchMesh1ToMesh2(params1, mesh1NumTriangles);
    overlapLauncher.launchMesh2ToMesh1(params2, mesh2NumTriangles);

    // Compact results
    // Allocate max output size (e.g. 2M to be safe, though true unique count is likely much lower)
    int max_output = 2000000; 
    MeshOverlapResult* d_merged_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshOverlapResult)));
    
    int numUnique = compact_hash_table(d_hash_table, hash_table_size, d_merged_results, max_output);
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
    }
    
    return {d_merged_results, numUnique};
}

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");
    
    std::string mesh1Path = "";
    std::string mesh2Path = "";
    std::string outputJsonPath = "mesh_overlap_timing.json";
    std::string ptxPath = detectPTXPath();
    int numberOfRuns = 1;
    bool exportResults = true;
    int warmupRuns = 2;
    float loadFactor = 0.75f;
    int manualHashSize = 0;
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--mesh1" && i + 1 < argc) {
                mesh1Path = argv[++i];
            }
            else if (arg == "--mesh2" && i + 1 < argc) {
                mesh2Path = argv[++i];
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
            else if (arg == "--load-factor" && i + 1 < argc) {
                loadFactor = std::stof(argv[++i]);
            }
            else if (arg == "--hash-size" && i + 1 < argc) {
                manualHashSize = std::atoi(argv[++i]);
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--mesh1 <path>] [--mesh2 <path>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mesh1 <path>         Path to first mesh dataset (geometry file)" << std::endl;
                std::cout << "  --mesh2 <path>         Path to second mesh dataset (geometry file)" << std::endl;
                std::cout << "  --output <path>        Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>        Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --warmup-runs <number> Number of warmup iterations" << std::endl;
                std::cout << "  --ptx <ptx_file>       Path to compiled PTX file (default: auto-detect)" << std::endl;
                std::cout << "  --load-factor <f>      Target load factor for hash table (default: 0.75)" << std::endl;
                std::cout << "  --hash-size <n>        Manually set hash table size (overrides estimation)" << std::endl;
                std::cout << "  --no-export            Do not export results to CSV" << std::endl;
                std::cout << "  --help, -h             Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "Mesh-to-Mesh Overlap Join" << std::endl;
    
    if (mesh1Path.empty()) {
        std::cerr << "Error: Mesh1 file path is required. Use --mesh1 <path_to_mesh1_file>" << std::endl;
        return 1;
    }
    
    if (mesh2Path.empty()) {
        std::cerr << "Error: Mesh2 file path is required. Use --mesh2 <path_to_mesh2_file>" << std::endl;
        return 1;
    }
    
    timer.next("Application Creation");
    
    OptixContext context;
    OptixPipelineManager basePipeline(context, ptxPath);
    MeshOverlapLauncher overlapLauncher(context, basePipeline);
    
    timer.next("Load Mesh1");
    std::cout << "Loading Mesh1 from: " << mesh1Path << std::endl;
    GeometryData mesh1Data = loadGeometryFromFile(mesh1Path);
    if (mesh1Data.vertices.empty()) {
        std::cerr << "Error: Failed to load Mesh1 from " << mesh1Path << std::endl;
        return 1;
    }
    std::cout << "Mesh1 loaded: " << mesh1Data.vertices.size() << " vertices, " 
              << mesh1Data.indices.size() << " triangles" << std::endl;
    
    timer.next("Load Mesh2");
    std::cout << "Loading Mesh2 from: " << mesh2Path << std::endl;
    GeometryData mesh2Data = loadGeometryFromFile(mesh2Path);
    if (mesh2Data.vertices.empty()) {
        std::cerr << "Error: Failed to load Mesh2 from " << mesh2Path << std::endl;
        return 1;
    }
    std::cout << "Mesh2 loaded: " << mesh2Data.vertices.size() << " vertices, " 
              << mesh2Data.indices.size() << " triangles" << std::endl;
    
    timer.next("Upload Mesh1");
    GeometryUploader mesh1Uploader;
    mesh1Uploader.upload(mesh1Data);
    std::cout << "Mesh1 uploaded to GPU" << std::endl;
    
    timer.next("Upload Mesh2");
    GeometryUploader mesh2Uploader;
    mesh2Uploader.upload(mesh2Data);
    std::cout << "Mesh2 uploaded to GPU" << std::endl;
    
    timer.next("Build Mesh1 Index");
    OptixAccelerationStructure mesh1AS(context, mesh1Uploader);
    mesh1AS.build();
    std::cout << "Mesh1 acceleration structure built" << std::endl;
    
    timer.next("Build Mesh2 Index");
    OptixAccelerationStructure mesh2AS(context, mesh2Uploader);
    mesh2AS.build();
    std::cout << "Mesh2 acceleration structure built" << std::endl;
    
    timer.next("Prepare Kernel Parameters");
    
    int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());
    
    // Determine Hash Table Size
    int hash_table_size = 16777216; // Default
    if (manualHashSize > 0) {
        hash_table_size = manualHashSize;
        std::cout << "Using manual hash table size: " << hash_table_size << std::endl;
    } else {
        std::cout << "Estimating selectivity..." << std::endl;
        
        // Calculate average triangles per object for better estimation
        // Use generic auto to support PinnedAllocator vectors
        auto getNumObjects = [](const auto& mapping) {
            if (mapping.empty()) return 1;
            int maxId = -1;
            // Iterate manually to avoid <algorithm> include dependency issues if not present, though main has it.
            for (int id : mapping) {
                if (id > maxId) maxId = id;
            }
            return maxId + 1;
        };

        int mesh1NumObjects = getNumObjects(mesh1Data.triangleToObject);
        int mesh2NumObjects = getNumObjects(mesh2Data.triangleToObject);
        
        float avgTris1 = (float)mesh1NumTriangles / std::max(1, mesh1NumObjects);
        float avgTris2 = (float)mesh2NumTriangles / std::max(1, mesh2NumObjects);
        
        size_t estimatedPairs = estimateSelectivityGPU(mesh1Data.eulerHistogram, mesh2Data.eulerHistogram);
        
        // Post-process logic: if we were in fallback mode (triangles), we must scale the result.
        // We know we are in fallback mode if object counts are empty.
        bool fallbackMode = mesh1Data.eulerHistogram.object_counts.empty() || mesh2Data.eulerHistogram.object_counts.empty();
        
        if (fallbackMode && estimatedPairs > 0) {
            std::cout << "  [Heuristic Scaling] Object counts missing. Scaling triangle pair estimate..." << std::endl;
            std::cout << "  Mesh1 Avg Tris/Obj: " << avgTris1 << std::endl;
            std::cout << "  Mesh2 Avg Tris/Obj: " << avgTris2 << std::endl;
            
            // Re-apply Heuristic: Object pairs ~ Raw Triangle Pairs / Min(AvgTrisA, AvgTrisB)
            double scalingFactor = 1.0 / std::min(avgTris1, avgTris2);
            size_t scaledEstimate = (size_t)(estimatedPairs * scalingFactor);
            if (scaledEstimate == 0) scaledEstimate = 1;
            
            std::cout << "  Scaling Factor: " << scalingFactor << std::endl;
            std::cout << "  Original Estimate (Triangle Pairs): " << estimatedPairs << std::endl;
            std::cout << "  Final Scaled Estimate (Object Pairs): " << scaledEstimate << std::endl;
            estimatedPairs = scaledEstimate;
        } else {
             std::cout << "  [Direct Estimation] Using result directly." << std::endl;
             std::cout << "  Final Estimate (Object Pairs): " << estimatedPairs << std::endl;
        }
        
        if (estimatedPairs > 0) {
            // Apply load factor
            size_t recommended = static_cast<size_t>(estimatedPairs / loadFactor);
            // Ensure some minimum sanity limits (e.g. 64K minimum)
            recommended = std::max((size_t)65536, recommended);
            
            size_t powerOf2 = 1;
            while (powerOf2 < recommended) powerOf2 <<= 1;
            hash_table_size = static_cast<int>(powerOf2);
             
            std::cout << "Final Hash Table Size (Load " << loadFactor << " + NextPow2): " << hash_table_size << std::endl;
        } else {
             std::cout << "Estimation unavailable (0 pairs or missing histograms). Using default: " << hash_table_size << std::endl;
        }
    }

    // Allocate Hash Table
    unsigned long long* d_hash_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));
    
    MeshOverlapLaunchParams params1;
    params1.mesh1_vertices = mesh1Uploader.getVertices();
    params1.mesh1_indices = mesh1Uploader.getIndices();
    params1.mesh1_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh2_handle = mesh2AS.getHandle();
    params1.mesh2_vertices = mesh2Uploader.getVertices();
    params1.mesh2_indices = mesh2Uploader.getIndices();
    params1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    // Legacy params set to null
    params1.collision_counts = nullptr;
    params1.collision_offsets = nullptr;
    params1.results = nullptr; 
    params1.pass = 0; 
    // Hash params
    params1.use_hash_table = true;
    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    
    MeshOverlapLaunchParams params2;
    params2.mesh1_vertices = mesh2Uploader.getVertices();
    params2.mesh1_indices = mesh2Uploader.getIndices();
    params2.mesh1_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params2.mesh1_num_triangles = mesh2NumTriangles;
    params2.mesh2_handle = mesh1AS.getHandle();
    params2.mesh2_vertices = mesh1Uploader.getVertices();
    params2.mesh2_indices = mesh1Uploader.getIndices();
    params2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    // Legacy params set to null
    params2.collision_counts = nullptr;
    params2.collision_offsets = nullptr;
    params2.results = nullptr; 
    params2.pass = 0;
    // Hash params
    params2.use_hash_table = true;
    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;
    
    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (hash-based)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            QueryResults warmupResults = executeHashQuery(
                overlapLauncher, params1, params2,
                mesh1NumTriangles, mesh2NumTriangles,
                d_hash_table, hash_table_size,
                false
            );
            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }
    
    timer.next("Query");
    
    std::cout << "\n=== Executing mesh overlap detection ===" << std::endl;
    QueryResults queryResults = executeHashQuery(
        overlapLauncher, params1, params2,
        mesh1NumTriangles, mesh2NumTriangles,
        d_hash_table, hash_table_size,
        true
    );
    
    MeshOverlapResult* d_merged_results = queryResults.d_merged_results;
    int numUnique = queryResults.numUnique;
    
    timer.next("GPU Deduplication");
    // Already done by hash table!
    std::cout << "Deduplication: Performed on-the-fly via Hash Table." << std::endl;
    
    timer.next("Download Results");
    
    std::vector<MeshOverlapResult> uniqueResults(numUnique);
    if (numUnique > 0) {
        CUDA_CHECK(cudaMemcpy(uniqueResults.data(), d_merged_results, 
                              numUnique * sizeof(MeshOverlapResult), 
                              cudaMemcpyDeviceToHost));
    }
    
    timer.next("Output");
    
    std::cout << "\n=== Mesh Overlap Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Unique object pairs: " << numUnique << std::endl;
    
    if (exportResults) {
        std::cout << "Exporting results to mesh_overlap_results.csv" << std::endl;
        std::ofstream csvFile("mesh_overlap_results.csv");
        csvFile << "object_id_mesh1,object_id_mesh2\n";
        for (const auto& result : uniqueResults) {
            csvFile << result.object_id_mesh1 << "," << result.object_id_mesh2 << "\n";
        }
        csvFile.close();
    }
    
    timer.next("Cleanup");
    
    if (d_merged_results) CUDA_CHECK(cudaFree(d_merged_results));
    CUDA_CHECK(cudaFree(d_hash_table));
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Mesh overlap join completed successfully" << std::endl;
    
    return 0;
}

