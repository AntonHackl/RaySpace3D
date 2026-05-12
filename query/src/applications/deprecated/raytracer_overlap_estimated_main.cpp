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
#include <unordered_map>
#include "../../optix/OptixContext.h"
#include "../../optix/OptixPipeline.h"
#include "../../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../../optix/OptixHelpers.h"
#include "../../raytracing/MeshOverlapEdgesLauncher.h"
#include "../../geometry/PrecomputedEdgeData.h"
#include "timer.h"
#include "ptx_utils.h"
#include "estimated_overlap.h"
#include "../app_cli_options.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
};

// Execute the overlap query using hash table deduplication
QueryResults executeHashQuery(
    MeshOverlapEdgesLauncher& edgesLauncher,
    MeshOverlapEdgesLaunchParams& edgesParams1,
    MeshOverlapEdgesLaunchParams& edgesParams2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    unsigned long long* d_hash_table,
    unsigned long long hash_table_size,
    long long estimated_pairs,
    PerformanceTimer* timer = nullptr,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    // Ensure params use hash table with bitwise optimisation (table size is power-of-two)
    edgesParams1.use_hash_table = true;
    edgesParams1.use_bitwise_hash = true;
    edgesParams1.hash_table = d_hash_table;
    edgesParams1.hash_table_size = hash_table_size;
    
    edgesParams2.use_hash_table = true;
    edgesParams2.use_bitwise_hash = true;
    edgesParams2.hash_table = d_hash_table;
    edgesParams2.hash_table_size = hash_table_size;

    auto t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh1ToMesh2(edgesParams1, mesh1NumEdges);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "Raytrace_Hash_Mesh1ToMesh2",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }

    t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh2ToMesh1(edgesParams2, mesh2NumEdges);
    t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "Raytrace_Hash_Mesh2ToMesh1",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }

    // Use estimated pairs to size the output buffer, with a fallback and a safety factor
    long long safe_estimate = (estimated_pairs > 0) ? (long long)(estimated_pairs * 1.2) : (long long)hash_table_size;
    // Ensure we don't allocate ridiculously small if estimate is off, utilize triangle count heuristic as floor
    long long triangle_heuristic = (long long)std::max(mesh1NumEdges, mesh2NumEdges) * 2;
    
    long long max_output_long = std::max(safe_estimate, triangle_heuristic);
    if (max_output_long < 2000000) max_output_long = 2000000;
    
    // Clamp to reasonable GPU memory limits if needed, but let's assume we have memory for now or let cudaMalloc fail
    // (Optional: clamp to hash_table_size as theoretical max unique items)
    if (max_output_long > hash_table_size) max_output_long = hash_table_size;
    
    int max_output = (int)max_output_long;

    MeshQueryResult* d_merged_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshQueryResult)));
    
    auto t_dedup_start = std::chrono::high_resolution_clock::now();
    int numUnique = compact_hash_table_pairs(d_hash_table, hash_table_size, d_merged_results, max_output);
    auto t_dedup_end = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "compact_hash_table_pairs",
            std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_end - t_dedup_start).count()
        );
    }
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
         if (numUnique >= max_output) {
             std::cerr << "WARNING: Output buffer full! Results may be truncated. Max output: " << max_output << std::endl;
         }
    }
    
    return {d_merged_results, numUnique};
}

// Helper to calculate next power of 2
unsigned long long nextPow2(unsigned long long v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

// Helper to calculate global average size of objects from grid statistics
float calculateGlobalAvgSize(const std::vector<SparseGridEntry>& sparseCells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            totalSize += (double)entry.stats.AvgSizeMean * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 0.0f;
    return (float)(totalSize / totalCount);
}

// Helper to calculate global average VolRatio from grid statistics
float calculateGlobalAvgVolRatio(const std::vector<SparseGridEntry>& sparseCells) {
    double totalRatio = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            totalRatio += (double)entry.stats.VolRatio * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 1.0f;  // Default to 1.0 (no correction)
    return (float)(totalRatio / totalCount);
}

class OverlapEstimatedCliOptions : public BenchmarkMeshPairCliOptions {
public:
    OverlapEstimatedCliOptions() : BenchmarkMeshPairCliOptions("estimated_overlap_timing.json") {}

    bool estimateOnly = false;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    int overlapMaxIterations = 100;

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(options);
        appendBenchmarkRunHelp(options);
        options.emplace_back("--gamma <float>", "Gamma parameter for estimation (default: 0.8)");
        options.emplace_back("--epsilon <float>", "Epsilon parameter for estimation (default: 0.001)");
        options.emplace_back("--overlap-max-iterations <int>", "Overlap ray iteration cap (default: 100)");
        options.emplace_back("--estimate-only", "Only run selectivity estimation, skip actual query");
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "--mesh1 <path> --mesh2 <path> [options]",
            "Overlap estimated query using selectivity estimation plus hash-based refinement.",
            options
        );
    }

protected:
    bool parseApplicationOption(const std::string& arg, int& i, int argc, char* argv[]) override {
        if (arg == "--gamma" && i + 1 < argc) {
            gamma = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--epsilon" && i + 1 < argc) {
            epsilon = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--estimate-only") {
            estimateOnly = true;
            return true;
        }
        if (arg == "--overlap-max-iterations" && i + 1 < argc) {
            overlapMaxIterations = std::stoi(argv[++i]);
            return true;
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    OverlapEstimatedCliOptions options;
    options.ptxPath = detectPTXPath();
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }

    options.sanitizeRunCounts();

    const std::string& mesh1Path = options.mesh1Path;
    const std::string& mesh2Path = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const std::string& ptxPath = options.ptxPath;
    const int numberOfRuns = options.numberOfRuns;
    const int warmupRuns = options.warmupRuns;
    const bool estimateOnly = options.estimateOnly;
    const float gamma = options.gamma;
    const float epsilon = options.epsilon;
    const int overlapMaxIterations = options.overlapMaxIterations;

    if (!options.hasRequiredMeshInputs()) {
        std::cerr << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
        return 1;
    }
    
    timer.start("Load Mesh1");
    GeometryData mesh1 = loadGeometryFromFile(mesh1Path);
    if (mesh1.vertices.empty()) {
        std::cerr << "Error loading mesh1." << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh1, mesh1Path, "Mesh1")) {
        return 1;
    }
    
    timer.next("Load Mesh2");
    GeometryData mesh2 = loadGeometryFromFile(mesh2Path);
    
    if (mesh2.vertices.empty()) {
        std::cerr << "Error loading mesh2." << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh2, mesh2Path, "Mesh2")) {
        return 1;
    }

    auto estimatePairs = [&](bool verbose) -> long long {
        long long estimatedPairs = 0;

        if (mesh1.grid.hasGrid && mesh2.grid.hasGrid) {
            if (std::abs(mesh1.grid.cellSize - mesh2.grid.cellSize) > 1e-5f) {
                if (verbose) {
                    std::cerr << "Warning: Grid cell sizes mismatch (Mesh1: " << mesh1.grid.cellSize 
                              << ", Mesh2: " << mesh2.grid.cellSize << "). Estimation may be invalid." << std::endl;
                }
            }

            float cellVolume = mesh1.grid.cellSize * mesh1.grid.cellSize * mesh1.grid.cellSize;

            struct Int3Hash {
                size_t operator()(const int3& k) const {
                    return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
                }
            };
            struct Int3Equal {
                bool operator()(const int3& a, const int3& b) const {
                    return a.x == b.x && a.y == b.y && a.z == b.z;
                }
            };

            std::unordered_map<int3, GridCell, Int3Hash, Int3Equal> mapA;
            for (const auto& entry : mesh1.grid.sparseCells) {
                mapA[entry.index] = entry.stats;
            }

            std::vector<GridCell> matchedA;
            std::vector<GridCell> matchedB;

            for (const auto& entry : mesh2.grid.sparseCells) {
                auto it = mapA.find(entry.index);
                if (it != mapA.end()) {
                    matchedA.push_back(it->second);
                    matchedB.push_back(entry.stats);
                }
            }

            int numMatchedCells = matchedA.size();

            float estimatedPairsFloat = 0.0f;
            if (numMatchedCells > 0) {
                estimatedPairsFloat = estimateOverlapSelectivity(
                    matchedA.data(),
                    matchedB.data(),
                    numMatchedCells,
                    cellVolume,
                    epsilon,
                    gamma
                );
            }

            float avgSize1 = calculateGlobalAvgSize(mesh1.grid.sparseCells);
            float avgSize2 = calculateGlobalAvgSize(mesh2.grid.sparseCells);
            float avgVolRatio1 = calculateGlobalAvgVolRatio(mesh1.grid.sparseCells);
            float avgVolRatio2 = calculateGlobalAvgVolRatio(mesh2.grid.sparseCells);

            float effectiveSize1 = avgSize1 * std::cbrt(avgVolRatio1);
            float effectiveSize2 = avgSize2 * std::cbrt(avgVolRatio2);

            float combinedSize = effectiveSize1 + effectiveSize2;
            float minkowskiVol = combinedSize * combinedSize * combinedSize;

            if (cellVolume < 1e-9f) cellVolume = 1e-9f;

            float alpha = minkowskiVol / cellVolume;
            if (alpha < 1.0f) alpha = 1.0f;

            estimatedPairs = (long long)(estimatedPairsFloat / alpha);

            if (verbose) {
                std::cout << "\n=== Selectivity Estimation (Overlap) ===" << std::endl;
                std::cout << "Matched Sparse Cells:      " << numMatchedCells << std::endl;
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
            std::cout << "Skipping estimation: Grid data not found." << std::endl;
        }

        return estimatedPairs;
    };

    auto computeHashTableSize = [](long long estimatedPairs) -> unsigned long long {
        unsigned long long hash_table_size = 16777216;
        if (estimatedPairs > 0) {
            unsigned long long target = (unsigned long long)(estimatedPairs / 0.5);
            if (target < 1024) target = 1024;
            if (target > 1073741824ULL) target = 1073741824ULL;
            hash_table_size = nextPow2(target);
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
    MeshOverlapEdgesLauncher edgesLauncher(context, basePipeline);

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

    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh1);
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh2);
    int mesh1NumEdges = mesh1EdgeData.num_edges;
    int mesh2NumEdges = mesh2EdgeData.num_edges;

    MeshOverlapEdgesLaunchParams edgesParams1 = {};
    edgesParams1.edge_starts = mesh1EdgeData.d_edge_starts;
    edgesParams1.edge_ends = mesh1EdgeData.d_edge_ends;
    edgesParams1.edge_source_object_ids = mesh1EdgeData.d_source_object_ids;
    edgesParams1.num_edges = mesh1NumEdges;
    edgesParams1.mesh2_handle = mesh2AS.getHandle();
    edgesParams1.mesh2_vertices = mesh2Uploader.getVertices();
    edgesParams1.mesh2_indices = (uint3*)mesh2Uploader.getIndices();
    edgesParams1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    edgesParams1.swap_pair_order = 0;
    edgesParams1.overlap_max_iterations = overlapMaxIterations;

    MeshOverlapEdgesLaunchParams edgesParams2 = {};
    edgesParams2.edge_starts = mesh2EdgeData.d_edge_starts;
    edgesParams2.edge_ends = mesh2EdgeData.d_edge_ends;
    edgesParams2.edge_source_object_ids = mesh2EdgeData.d_source_object_ids;
    edgesParams2.num_edges = mesh2NumEdges;
    edgesParams2.mesh2_handle = mesh1AS.getHandle();
    edgesParams2.mesh2_vertices = mesh1Uploader.getVertices();
    edgesParams2.mesh2_indices = (uint3*)mesh1Uploader.getIndices();
    edgesParams2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    edgesParams2.swap_pair_order = 1;
    edgesParams2.overlap_max_iterations = overlapMaxIterations;

    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (estimation + hash query)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            long long warmupEstimatedPairs = estimatePairs(false);
            unsigned long long warmupHashSize = computeHashTableSize(warmupEstimatedPairs);
            unsigned long long* d_warmup_hash_table = nullptr;
            CUDA_CHECK(cudaMalloc(&d_warmup_hash_table, warmupHashSize * sizeof(unsigned long long)));

            QueryResults warmupResults = executeHashQuery(
                edgesLauncher,
                edgesParams1,
                edgesParams2,
                mesh1NumEdges,
                mesh2NumEdges,
                d_warmup_hash_table,
                warmupHashSize,
                warmupEstimatedPairs,
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
            std::cout << "Using Power-of-Two Hash Table Size (bitwise opt): " << hash_table_size << std::endl;
        }

        unsigned long long* d_hash_table = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

        timer.next("Execute Hash Query");
        QueryResults queryResults = executeHashQuery(
            edgesLauncher,
            edgesParams1,
            edgesParams2,
            mesh1NumEdges,
            mesh2NumEdges,
            d_hash_table,
            hash_table_size,
            estimatedPairs,
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
    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);

    std::set<int> mesh1UniqueObjects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    int mesh1NumObjects = mesh1UniqueObjects.size();
    std::set<int> mesh2UniqueObjects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh2NumObjects = mesh2UniqueObjects.size();

    std::cout << "\n=== Mesh Overlap Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh1 objects: " << mesh1NumObjects << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Mesh2 objects: " << mesh2NumObjects << std::endl;

    std::cout << "Unique object pairs: " << finalNumUnique << std::endl;

    timer.finish(outputJsonPath);
    
    std::cout << "\nQuery completed in " << (double)timer.getTotalDuration() / 1000.0 << " ms." << std::endl;
    std::cout << "Results saved to: " << outputJsonPath << std::endl;

    return 0;
}
