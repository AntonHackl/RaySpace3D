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
#include <cmath>
#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_intersection.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshIntersectionLauncher.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_intersection.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
};

// Execute the intersection query using hash table deduplication
QueryResults executeHashQuery(
    MeshIntersectionLauncher& intersectionLauncher,
    MeshIntersectionLaunchParams& params1,
    MeshIntersectionLaunchParams& params2,
    int mesh1NumTriangles,
    int mesh2NumTriangles,
    unsigned long long* d_hash_table,
    int hash_table_size,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    params1.use_hash_table = true;
    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    
    params2.use_hash_table = true;
    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;

    intersectionLauncher.launchMesh1ToMesh2(params1, mesh1NumTriangles);
    intersectionLauncher.launchMesh2ToMesh1(params2, mesh2NumTriangles);

    int max_output = hash_table_size; 

    MeshQueryResult* d_merged_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshQueryResult)));
    
    int numUnique = compact_hash_table_pairs(d_hash_table, hash_table_size, d_merged_results, max_output);
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
    }
    
    return {d_merged_results, numUnique};
}


// Helper to calculate global average size of objects from grid statistics
float calculateGlobalAvgSize(const std::vector<GridCell>& cells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& cell : cells) {
        if (cell.TouchCount > 0) {
            // Un-average to get sum of sizes in this cell
            totalSize += (double)cell.AvgSizeMean * (double)cell.TouchCount;
            totalCount += cell.TouchCount;
        }
    }
    
    if (totalCount == 0) return 0.0f;
    return (float)(totalSize / totalCount);
}

// Helper to calculate global average VolRatio from grid statistics
float calculateGlobalAvgVolRatio(const std::vector<GridCell>& cells) {
    double totalRatio = 0.0;
    long long totalCount = 0;
    
    for (const auto& cell : cells) {
        if (cell.TouchCount > 0) {
            totalRatio += (double)cell.VolRatio * (double)cell.TouchCount;
            totalCount += cell.TouchCount;
        }
    }
    
    if (totalCount == 0) return 1.0f;  // Default to 1.0 (no correction)
    return (float)(totalRatio / totalCount);
}

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    
    std::string mesh1Path = "";
    std::string mesh2Path = "";
    std::string outputJsonPath = "estimated_intersection_timing.json";
    std::string ptxPath = detectPTXPath();
    bool estimateOnly = false;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    
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
            else if (arg == "--gamma" && i + 1 < argc) {
                gamma = std::stof(argv[++i]);
            }
            else if (arg == "--epsilon" && i + 1 < argc) {
                epsilon = std::stof(argv[++i]);
            }
            else if (arg == "--estimate-only") {
                estimateOnly = true;
            }
        }
    }
    
    if (mesh1Path.empty() || mesh2Path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
        return 1;
    }
    
    timer.start("Load Mesh1");
    GeometryData mesh1 = loadGeometryFromFile(mesh1Path);
    if (mesh1.vertices.empty()) {
        std::cerr << "Error loading mesh1." << std::endl;
        return 1;
    }
    
    timer.next("Load Mesh2");
    GeometryData mesh2 = loadGeometryFromFile(mesh2Path);
    
    if (mesh2.vertices.empty()) {
        std::cerr << "Error loading mesh2." << std::endl;
        return 1;
    }

    // --- ESTIMATION PHASE ---
    timer.next("Selectivity Estimation");
    
    long long estimatedPairs = 0;

    if (mesh1.grid.hasGrid && mesh2.grid.hasGrid) {
        // Validation
        if (mesh1.grid.resolution.x != mesh2.grid.resolution.x ||
            mesh1.grid.minBound.x != mesh2.grid.minBound.x ||
            mesh1.grid.maxBound.x != mesh2.grid.maxBound.x) { // Basic check
            std::cerr << "Warning: Grid parameters (resolution/bounds) mismatch. Estimation may be invalid." << std::endl;
        }

        float width = mesh1.grid.maxBound.x - mesh1.grid.minBound.x;
        float height = mesh1.grid.maxBound.y - mesh1.grid.minBound.y;
        float depth = mesh1.grid.maxBound.z - mesh1.grid.minBound.z;
        float csX = width / mesh1.grid.resolution.x;
        float csY = height / mesh1.grid.resolution.y;
        float csZ = depth / mesh1.grid.resolution.z;
        float cellVolume = csX * csY * csZ;
        
        int numCells = mesh1.grid.resolution.x * mesh1.grid.resolution.y * mesh1.grid.resolution.z;
        if (mesh1.grid.cells.size() == numCells && mesh2.grid.cells.size() == numCells) {
            float estimatedPairsFloat = estimateIntersectionSelectivity(
                mesh1.grid.cells.data(), 
                mesh2.grid.cells.data(), 
                numCells, 
                cellVolume,
                epsilon,
                gamma
            );
            
            // --- Normalization (Replication Correction) ---
            float avgSize1 = calculateGlobalAvgSize(mesh1.grid.cells);
            float avgSize2 = calculateGlobalAvgSize(mesh2.grid.cells);
            float avgVolRatio1 = calculateGlobalAvgVolRatio(mesh1.grid.cells);
            float avgVolRatio2 = calculateGlobalAvgVolRatio(mesh2.grid.cells);
            
            // Scale sizes by cube root of VolRatio to get "effective" linear dimension
            // For sparse/elongated objects, this reduces the effective size significantly
            float effectiveSize1 = avgSize1 * std::cbrt(avgVolRatio1);
            float effectiveSize2 = avgSize2 * std::cbrt(avgVolRatio2);
            
            // Calculate alpha: Volume of Minkowski Sum in grid cell units
            // Using effective sizes to account for object shapes
            float combinedSize = effectiveSize1 + effectiveSize2;
            float minkowskiVol = combinedSize * combinedSize * combinedSize;
            
            if (cellVolume < 1e-9f) cellVolume = 1e-9f;
            
            float alpha = minkowskiVol / cellVolume;
            if (alpha < 1.0f) alpha = 1.0f; // Cannot be less than 1 cell
            
            estimatedPairs = (long long)(estimatedPairsFloat / alpha);

            std::cout << "\n=== Selectivity Estimation ===" << std::endl;
            std::cout << "Raw Potential Pairs:       " << (long long)estimatedPairsFloat << std::endl;
            std::cout << "Avg Object Size (Mesh1):   " << avgSize1 << std::endl;
            std::cout << "Avg Object Size (Mesh2):   " << avgSize2 << std::endl;
            std::cout << "Avg VolRatio (Mesh1):      " << avgVolRatio1 << std::endl;
            std::cout << "Avg VolRatio (Mesh2):      " << avgVolRatio2 << std::endl;
            std::cout << "Effective Size (Mesh1):    " << effectiveSize1 << std::endl;
            std::cout << "Effective Size (Mesh2):    " << effectiveSize2 << std::endl;
            std::cout << "Replication Factor (alpha):" << alpha << std::endl;
            std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
            std::cout << "==============================\n" << std::endl;
        } else {
             std::cerr << "Error: Grid has " << mesh1.grid.cells.size() << "/" << mesh2.grid.cells.size() 
                       << " cells, expected " << numCells << ". Skipping estimation." << std::endl;
        }
    } else {
        std::cout << "Skipping estimation: Grid data not found in one or both datasets." << std::endl;
        std::cout << "Run preprocess_dataset with --generate-grid to enable estimation." << std::endl;
    }

    if (estimateOnly) {
        timer.finish(outputJsonPath);
        return 0;
    }

    int hash_table_size = 16777216;
    if (estimatedPairs > 0) {
        unsigned long long target = (unsigned long long)(estimatedPairs / 0.5);
        if (target < 1024) target = 1024;
        
        // Cap to reasonable int size for hash table param
        if (target > 1073741824ULL) target = 1073741824ULL;
        hash_table_size = (int)target;
        if (hash_table_size % 2 == 0) {
            hash_table_size += 1;
        }
    }

    std::cout << "\n=== Query Configuration ===" << std::endl;
    std::cout << "Estimated Pairs:    " << estimatedPairs << std::endl;
    std::cout << "Hash Table Size:    " << hash_table_size << " (Load Factor ~0.5)" << std::endl;
    std::cout << "===========================\n" << std::endl;

    // --- EXECUTION PHASE ---
    timer.next("Init OptiX");

    // Create OptiX context and pipeline (reuse existing project patterns)
    OptixContext context;
    OptixPipelineManager basePipeline(context, ptxPath);
    MeshIntersectionLauncher intersectionLauncher(context, basePipeline);

    timer.next("Upload Mesh1");

    GeometryUploader mesh1Uploader;
    mesh1Uploader.upload(mesh1);

    timer.next("Upload Mesh2");
    GeometryUploader mesh2Uploader;
    mesh2Uploader.upload(mesh2);

    timer.next("Build Mesh1 Index");
    OptixAccelerationStructure mesh1AS(context, mesh1Uploader);
    mesh1AS.build();

    timer.next("Build Mesh2 Index");
    OptixAccelerationStructure mesh2AS(context, mesh2Uploader);
    mesh2AS.build();

    timer.next("Prepare Kernel Parameters");

    int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());

    // Allocate Hash Table
    unsigned long long* d_hash_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

    // Allocate object tracking buffers
    std::set<int> mesh1Objects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    std::set<int> mesh2Objects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh1NumObjects = mesh1Objects.size();
    int mesh2NumObjects = mesh2Objects.size();

    unsigned char* d_object_tested1 = nullptr;
    unsigned char* d_object_tested2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_object_tested1, mesh1NumObjects * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_object_tested2, mesh2NumObjects * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_object_tested1, 0, mesh1NumObjects * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(d_object_tested2, 0, mesh2NumObjects * sizeof(unsigned char)));

    MeshIntersectionLaunchParams params1;
    params1.mesh1_vertices = mesh1Uploader.getVertices();
    params1.mesh1_indices = mesh1Uploader.getIndices();
    params1.mesh1_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh1_num_objects = mesh1NumObjects;
    params1.mesh2_handle = mesh2AS.getHandle();
    params1.mesh2_vertices = mesh2Uploader.getVertices();
    params1.mesh2_indices = mesh2Uploader.getIndices();
    params1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params1.mesh2_num_objects = mesh2NumObjects;

    MeshIntersectionLaunchParams params2;
    params2.mesh1_vertices = mesh2Uploader.getVertices();
    params2.mesh1_indices = mesh2Uploader.getIndices();
    params2.mesh1_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params2.mesh1_num_triangles = mesh2NumTriangles;
    params2.mesh1_num_objects = mesh2NumObjects;
    params2.mesh2_handle = mesh1AS.getHandle();
    params2.mesh2_vertices = mesh1Uploader.getVertices();
    params2.mesh2_indices = mesh1Uploader.getIndices();
    params2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params2.mesh2_num_objects = mesh1NumObjects;

    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    params1.use_hash_table = true;

    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;
    params2.use_hash_table = true;

    params1.object_tested = d_object_tested1;
    params2.object_tested = d_object_tested2;

    timer.next("Query");
    std::cout << "Running Intersection Query..." << std::endl;

    QueryResults results = executeHashQuery(
        intersectionLauncher, params1, params2,
        mesh1NumTriangles, mesh2NumTriangles,
        d_hash_table, hash_table_size
    );

    std::cout << "Actual Intersection Pairs: " << results.numUnique << std::endl;

    timer.next("Cleanup");
    CUDA_CHECK(cudaFree(d_hash_table));
    CUDA_CHECK(cudaFree(results.d_merged_results));
    CUDA_CHECK(cudaFree(d_object_tested1));
    CUDA_CHECK(cudaFree(d_object_tested2));

    mesh1Uploader.free();
    mesh2Uploader.free();

    timer.finish(outputJsonPath);
    return 0;
}
