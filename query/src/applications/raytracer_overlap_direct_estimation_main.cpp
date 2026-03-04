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
#include <chrono>
#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_overlap.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshOverlapLauncher.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_overlap.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
};

enum class QueryDirection {
    Both,
    Mesh1ToMesh2,
    Mesh2ToMesh1
};

static const char* directionToString(QueryDirection direction) {
    switch (direction) {
        case QueryDirection::Mesh1ToMesh2: return "mesh1_to_mesh2";
        case QueryDirection::Mesh2ToMesh1: return "mesh2_to_mesh1";
        case QueryDirection::Both:
        default: return "both";
    }
}

static bool parseDirection(const std::string& raw, QueryDirection& outDirection) {
    if (raw == "both") {
        outDirection = QueryDirection::Both;
        return true;
    }
    if (raw == "mesh1_to_mesh2") {
        outDirection = QueryDirection::Mesh1ToMesh2;
        return true;
    }
    if (raw == "mesh2_to_mesh1") {
        outDirection = QueryDirection::Mesh2ToMesh1;
        return true;
    }
    return false;
}

// Execute the overlap query using hash table deduplication
QueryResults executeHashQuery(
    MeshOverlapLauncher& overlapLauncher,
    MeshOverlapLaunchParams& params1,
    MeshOverlapLaunchParams& params2,
    int mesh1NumTriangles,
    int mesh2NumTriangles,
    unsigned long long* d_hash_table,
    unsigned long long hash_table_size,
    long long estimated_pairs,
    QueryDirection direction,
    PerformanceTimer* timer = nullptr,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    // Ensure params use hash table (no bitwise opt - table size is not power-of-two)
    params1.use_hash_table = true;
    params1.use_bitwise_hash = false;
    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    
    params2.use_hash_table = true;
    params2.use_bitwise_hash = false;
    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;

    if (direction == QueryDirection::Both || direction == QueryDirection::Mesh1ToMesh2) {
        auto t0 = std::chrono::high_resolution_clock::now();
        overlapLauncher.launchMesh1ToMesh2(params1, mesh1NumTriangles);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (direction == QueryDirection::Both || direction == QueryDirection::Mesh2ToMesh1) {
        auto t0 = std::chrono::high_resolution_clock::now();
        overlapLauncher.launchMesh2ToMesh1(params2, mesh2NumTriangles);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    // Use estimated pairs to size the output buffer, with a fallback and a safety factor
    long long safe_estimate = (estimated_pairs > 0) ? (long long)(estimated_pairs * 1.2) : (long long)hash_table_size;
    // Ensure we don't allocate ridiculously small if estimate is off, utilize triangle count heuristic as floor
    long long triangle_heuristic = (long long)std::max(mesh1NumTriangles, mesh2NumTriangles) * 2;
    
    long long max_output_long = std::max(safe_estimate, triangle_heuristic);
    if (max_output_long < 2000000) max_output_long = 2000000;
    
    // Clamp to reasonable GPU memory limits if needed, but let's assume we have memory for now or let cudaMalloc fail
    // (Optional: clamp to hash_table_size as theoretical max unique items)
    if (max_output_long > hash_table_size) max_output_long = hash_table_size;
    
    int max_output = (int)max_output_long;

    MeshQueryResult* d_merged_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshQueryResult)));
    
    int numUnique = compact_hash_table_pairs(d_hash_table, hash_table_size, d_merged_results, max_output);
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
         if (numUnique >= max_output) {
             std::cerr << "WARNING: Output buffer full! Results may be truncated. Max output: " << max_output << std::endl;
         }
    }
    
    return {d_merged_results, numUnique};
}

// Helper to calculate global average size of objects from grid statistics
float calculateGlobalAvgSize(const std::vector<GridCell>& cells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& cell : cells) {
        if (cell.TouchCount > 0) {
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
    std::string outputJsonPath = "estimated_overlap_timing.json";
    std::string ptxPath = detectPTXPath();
    int numberOfRuns = 1;
    int warmupRuns = 2;
    bool estimateOnly = false;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    QueryDirection queryDirection = QueryDirection::Both;
    std::string pairsOutputPath = "";
    
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mesh1 <path>         Path to first mesh dataset (geometry file)" << std::endl;
                std::cout << "  --mesh2 <path>         Path to second mesh dataset (geometry file)" << std::endl;
                std::cout << "  --output <path>        Path to JSON file for performance timing output (default: estimated_overlap_timing.json)" << std::endl;
                std::cout << "  --ptx <ptx_file>       Path to compiled PTX file (default: auto-detect)" << std::endl;
                std::cout << "  --runs <number>        Number of measured query runs (default: 1)" << std::endl;
                std::cout << "  --warmup-runs <number> Number of warmup iterations (default: 2)" << std::endl;
                std::cout << "  --gamma <float>        Gamma parameter for estimation (default: 0.8)" << std::endl;
                std::cout << "  --epsilon <float>      Epsilon parameter for estimation (default: 0.001)" << std::endl;
                std::cout << "  --query-direction <d>  Query direction: both|mesh1_to_mesh2|mesh2_to_mesh1 (default: both)" << std::endl;
                std::cout << "  --pairs-output <path>  Optional CSV export path for unique result pairs" << std::endl;
                std::cout << "  --estimate-only        Only run selectivity estimation, skip actual query" << std::endl;
                std::cout << "  --help, -h             Show this help message" << std::endl;
                return 0;
            }
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
            else if (arg == "--gamma" && i + 1 < argc) {
                gamma = std::stof(argv[++i]);
            }
            else if (arg == "--epsilon" && i + 1 < argc) {
                epsilon = std::stof(argv[++i]);
            }
            else if (arg == "--query-direction" && i + 1 < argc) {
                std::string directionRaw = argv[++i];
                if (!parseDirection(directionRaw, queryDirection)) {
                    std::cerr << "Error: Invalid --query-direction value: " << directionRaw
                              << ". Expected one of: both, mesh1_to_mesh2, mesh2_to_mesh1" << std::endl;
                    return 1;
                }
            }
            else if (arg == "--pairs-output" && i + 1 < argc) {
                pairsOutputPath = argv[++i];
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

    if (numberOfRuns < 1) numberOfRuns = 1;
    if (warmupRuns < 0) warmupRuns = 0;
    
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

    auto estimatePairs = [&](bool verbose) -> long long {
        long long estimatedPairs = 0;

        if (mesh1.grid.hasGrid && mesh2.grid.hasGrid) {
            float width = mesh1.grid.maxBound.x - mesh1.grid.minBound.x;
            float height = mesh1.grid.maxBound.y - mesh1.grid.minBound.y;
            float depth = mesh1.grid.maxBound.z - mesh1.grid.minBound.z;
            float csX = width / mesh1.grid.resolution.x;
            float csY = height / mesh1.grid.resolution.y;
            float csZ = depth / mesh1.grid.resolution.z;
            float cellVolume = csX * csY * csZ;

            int numCells = mesh1.grid.resolution.x * mesh1.grid.resolution.y * mesh1.grid.resolution.z;
            if (mesh1.grid.cells.size() == (size_t)numCells && mesh2.grid.cells.size() == (size_t)numCells) {
                float estimatedPairsFloat = estimateOverlapSelectivity(
                    mesh1.grid.cells.data(),
                    mesh2.grid.cells.data(),
                    numCells,
                    cellVolume,
                    epsilon,
                    gamma
                );

                float avgSize1 = calculateGlobalAvgSize(mesh1.grid.cells);
                float avgSize2 = calculateGlobalAvgSize(mesh2.grid.cells);
                float avgVolRatio1 = calculateGlobalAvgVolRatio(mesh1.grid.cells);
                float avgVolRatio2 = calculateGlobalAvgVolRatio(mesh2.grid.cells);

                float effectiveSize1 = avgSize1 * std::cbrt(avgVolRatio1);
                float effectiveSize2 = avgSize2 * std::cbrt(avgVolRatio2);

                float combinedSize = effectiveSize1 + effectiveSize2;
                float minkowskiVol = combinedSize * combinedSize * combinedSize;

                if (cellVolume < 1e-9f) cellVolume = 1e-9f;

                float alpha = minkowskiVol / cellVolume;
                if (alpha < 1.0f) alpha = 1.0f;

                estimatedPairs = (long long)(estimatedPairsFloat / alpha);

                if (verbose) {
                    std::cout << "\n=== Selectivity Estimation (Overlap - Direct) ===" << std::endl;
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
                }
            } else if (verbose) {
                std::cerr << "Error: Grid mismatch. Skipping estimation." << std::endl;
            }
        } else if (verbose) {
            std::cout << "Skipping estimation: Grid data not found." << std::endl;
        }

        return estimatedPairs;
    };

    auto computeHashTableSize = [](long long estimatedPairs) -> unsigned long long {
        unsigned long long hash_table_size = 16777216;
        if (estimatedPairs > 0) {
            unsigned long long target = (unsigned long long)(estimatedPairs / 0.5);
            if (target < 1024) target = 1024;
            if (target > 2147483648ULL) target = 2147483648ULL;

            hash_table_size = target;
            if (hash_table_size % 2 == 0) hash_table_size++;
        }
        return hash_table_size;
    };

    if (estimateOnly) {
        timer.next("Selectivity Estimation");
        (void)estimatePairs(true);
        timer.finish(outputJsonPath);
        return 0;
    }

    // --- EXECUTION PHASE ---
    timer.next("Init OptiX");
    OptixContext context;
    OptixPipelineManager basePipeline(context, ptxPath);
    MeshOverlapLauncher overlapLauncher(context, basePipeline);

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

    MeshOverlapLaunchParams params1 = {};
    params1.mesh1_vertices = mesh1Uploader.getVertices();
    params1.mesh1_indices = (uint3*)mesh1Uploader.getIndices();
    params1.mesh1_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh2_handle = mesh2AS.getHandle();
    params1.mesh2_vertices = mesh2Uploader.getVertices();
    params1.mesh2_indices = (uint3*)mesh2Uploader.getIndices();
    params1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();

    MeshOverlapLaunchParams params2 = {};
    params2.mesh1_vertices = mesh2Uploader.getVertices();
    params2.mesh1_indices = (uint3*)mesh2Uploader.getIndices();
    params2.mesh1_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params2.mesh1_num_triangles = mesh2NumTriangles;
    params2.mesh2_handle = mesh1AS.getHandle();
    params2.mesh2_vertices = mesh1Uploader.getVertices();
    params2.mesh2_indices = (uint3*)mesh1Uploader.getIndices();
    params2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();

    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (estimation + hash query)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            long long warmupEstimatedPairs = estimatePairs(false);
            unsigned long long warmupHashSize = computeHashTableSize(warmupEstimatedPairs);
            unsigned long long* d_warmup_hash_table = nullptr;
            CUDA_CHECK(cudaMalloc(&d_warmup_hash_table, warmupHashSize * sizeof(unsigned long long)));

            QueryResults warmupResults = executeHashQuery(
                overlapLauncher,
                params1,
                params2,
                mesh1NumTriangles,
                mesh2NumTriangles,
                d_warmup_hash_table,
                warmupHashSize,
                warmupEstimatedPairs,
                queryDirection,
                nullptr,
                false
            );

            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
            CUDA_CHECK(cudaFree(d_warmup_hash_table));
        }
    }

    int finalNumUnique = 0;
    std::vector<MeshQueryResult> hostResults;
    for (int run = 0; run < numberOfRuns; ++run) {
        bool verboseRun = (run == 0);

        timer.next("Selectivity Estimation");
        long long estimatedPairs = estimatePairs(verboseRun);
        unsigned long long hash_table_size = computeHashTableSize(estimatedPairs);

        if (verboseRun) {
            std::cout << "Using Direct Estimated Hash Table Size: " << hash_table_size << std::endl;
        }

        unsigned long long* d_hash_table = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

        timer.next("Execute Hash Query");
        QueryResults queryResults = executeHashQuery(
            overlapLauncher,
            params1,
            params2,
            mesh1NumTriangles,
            mesh2NumTriangles,
            d_hash_table,
            hash_table_size,
            estimatedPairs,
            queryDirection,
            &timer,
            verboseRun
        );

        timer.next("Download Results");
        hostResults.clear();
        if (queryResults.numUnique > 0) {
            hostResults.resize(queryResults.numUnique);
            CUDA_CHECK(cudaMemcpy(hostResults.data(), queryResults.d_merged_results,
                                  (size_t)queryResults.numUnique * sizeof(MeshQueryResult),
                                  cudaMemcpyDeviceToHost));
        }
        finalNumUnique = queryResults.numUnique;

        if (queryResults.d_merged_results) CUDA_CHECK(cudaFree(queryResults.d_merged_results));
        CUDA_CHECK(cudaFree(d_hash_table));
    }

    timer.next("Cleanup");

    std::set<int> mesh1UniqueObjects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    int mesh1NumObjects = mesh1UniqueObjects.size();
    std::set<int> mesh2UniqueObjects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh2NumObjects = mesh2UniqueObjects.size();

    std::cout << "\n=== Mesh Overlap Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh1 objects: " << mesh1NumObjects << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Mesh2 objects: " << mesh2NumObjects << std::endl;
    std::cout << "Mesh1 Universe Min: [" << mesh1.grid.minBound.x << ", " << mesh1.grid.minBound.y << ", " << mesh1.grid.minBound.z << "]" << std::endl;
    std::cout << "Mesh1 Universe Max: [" << mesh1.grid.maxBound.x << ", " << mesh1.grid.maxBound.y << ", " << mesh1.grid.maxBound.z << "]" << std::endl;
    std::cout << "Mesh2 Universe Min: [" << mesh2.grid.minBound.x << ", " << mesh2.grid.minBound.y << ", " << mesh2.grid.minBound.z << "]" << std::endl;
    std::cout << "Mesh2 Universe Max: [" << mesh2.grid.maxBound.x << ", " << mesh2.grid.maxBound.y << ", " << mesh2.grid.maxBound.z << "]" << std::endl;
    std::cout << "Query direction: " << directionToString(queryDirection) << std::endl;
    std::cout << "Unique object pairs: " << finalNumUnique << std::endl;

    if (!pairsOutputPath.empty()) {
        std::ofstream csvFile(pairsOutputPath);
        if (!csvFile.is_open()) {
            std::cerr << "Warning: Failed to open pairs output file: " << pairsOutputPath << std::endl;
        } else {
            csvFile << "object_id_mesh1,object_id_mesh2\n";
            for (const auto& result : hostResults) {
                csvFile << result.object_id_mesh1 << "," << result.object_id_mesh2 << "\n";
            }
            csvFile.close();
            std::cout << "Pair results written to: " << pairsOutputPath << std::endl;
        }
    }

    timer.finish(outputJsonPath);
    
    std::cout << "\nQuery completed in " << (double)timer.getTotalDuration() / 1000.0 << " ms." << std::endl;
    std::cout << "Results saved to: " << outputJsonPath << std::endl;

    return 0;
}
