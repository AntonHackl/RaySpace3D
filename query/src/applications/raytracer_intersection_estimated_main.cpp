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
#include <stdexcept>
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
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_intersection.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
};

enum class QueryDirection {
    Both,
    Mesh1ToMesh2,
    Mesh2ToMesh1
};

QueryDirection parseQueryDirection(const std::string& direction) {
    if (direction == "both") {
        return QueryDirection::Both;
    }
    if (direction == "mesh1_to_mesh2") {
        return QueryDirection::Mesh1ToMesh2;
    }
    if (direction == "mesh2_to_mesh1") {
        return QueryDirection::Mesh2ToMesh1;
    }
    throw std::invalid_argument("Invalid query direction: " + direction);
}

int parseContainmentQueryPointMode(const std::string& mode) {
    if (mode == "vertex") {
        return 0;
    }
    if (mode == "centroid") {
        return 1;
    }
    throw std::invalid_argument("Invalid containment query point mode: " + mode);
}

// Execute the intersection query using hash table deduplication
QueryResults executeHashQuery(
    MeshIntersectionLauncher& intersectionLauncher,
    MeshIntersectionLaunchParams& params1,
    MeshIntersectionLaunchParams& params2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    int mesh1NumObjects,
    int mesh2NumObjects,
    unsigned long long* d_hash_table,
    int hash_table_size,
    QueryDirection queryDirection,
    PerformanceTimer* timer = nullptr,
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

    const bool runMesh1ToMesh2 = (queryDirection == QueryDirection::Both || queryDirection == QueryDirection::Mesh1ToMesh2);
    const bool runMesh2ToMesh1 = (queryDirection == QueryDirection::Both || queryDirection == QueryDirection::Mesh2ToMesh1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    if (runMesh1ToMesh2) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchOverlapMesh1ToMesh2(params1, mesh1NumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh2ToMesh1) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchOverlapMesh2ToMesh1(params2, mesh2NumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh1ToMesh2) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh1ToMesh2(params1, mesh1NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh2ToMesh1) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh2ToMesh1(params2, mesh2NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    int max_output = hash_table_size; 

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
    std::string queryDirectionArg = "both";
    std::string containmentQueryPointArg = "vertex";
    bool estimateOnly = false;
    bool enableProfilingStats = false;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    float hashLoadFactor = 0.5f;
    int overlapMaxIterations = 100;
    int containmentMaxIterations = 512;
    
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
            else if (arg == "--query-direction" && i + 1 < argc) {
                queryDirectionArg = argv[++i];
            }
            else if (arg == "--containment-query-point" && i + 1 < argc) {
                containmentQueryPointArg = argv[++i];
            }
            else if (arg == "--overlap-max-iterations" && i + 1 < argc) {
                overlapMaxIterations = std::stoi(argv[++i]);
            }
            else if (arg == "--containment-max-iterations" && i + 1 < argc) {
                containmentMaxIterations = std::stoi(argv[++i]);
            }
            else if (arg == "--hash-load-factor" && i + 1 < argc) {
                hashLoadFactor = std::stof(argv[++i]);
            }
            else if (arg == "--enable-profiling-stats") {
                enableProfilingStats = true;
            }
        }
    }

    QueryDirection queryDirection = QueryDirection::Both;
    int containmentQueryPointMode = 0;
    try {
        queryDirection = parseQueryDirection(queryDirectionArg);
        containmentQueryPointMode = parseContainmentQueryPointMode(containmentQueryPointArg);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << "Valid values for --query-direction are: both, mesh1_to_mesh2, mesh2_to_mesh1" << std::endl;
        std::cerr << "Valid values for --containment-query-point are: vertex, centroid" << std::endl;
        return 1;
    }

    if (hashLoadFactor <= 0.0f || hashLoadFactor > 1.0f) {
        std::cerr << "Invalid --hash-load-factor. Expected value in (0, 1]." << std::endl;
        return 1;
    }
    if (overlapMaxIterations <= 0 || containmentMaxIterations <= 0) {
        std::cerr << "Iteration limits must be > 0." << std::endl;
        return 1;
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
        unsigned long long target = (unsigned long long)(estimatedPairs / hashLoadFactor);
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
    std::cout << "Hash Table Size:    " << hash_table_size << " (Load Factor ~" << hashLoadFactor << ")" << std::endl;
    std::cout << "Query Direction:    " << queryDirectionArg << std::endl;
    std::cout << "Containment Point:  " << containmentQueryPointArg << std::endl;
    std::cout << "Overlap Max Iter:   " << overlapMaxIterations << std::endl;
    std::cout << "Contain Max Iter:   " << containmentMaxIterations << std::endl;
    std::cout << "Profiling Stats:    " << (enableProfilingStats ? "enabled" : "disabled") << std::endl;
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
    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh1);
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh2);
    int mesh1NumEdges = mesh1EdgeData.num_edges;
    int mesh2NumEdges = mesh2EdgeData.num_edges;

    // Allocate Hash Table
    unsigned long long* d_hash_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

    MeshIntersectionProfilingStats* d_profiling_stats = nullptr;
    MeshIntersectionProfilingStats h_profiling_stats = {};
    if (enableProfilingStats) {
        CUDA_CHECK(cudaMalloc(&d_profiling_stats, sizeof(MeshIntersectionProfilingStats)));
        CUDA_CHECK(cudaMemset(d_profiling_stats, 0, sizeof(MeshIntersectionProfilingStats)));
    }

    // Allocate object tracking buffers
    std::set<int> mesh1Objects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    std::set<int> mesh2Objects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh1NumObjects = mesh1Objects.size();
    int mesh2NumObjects = mesh2Objects.size();

    std::vector<int> firstTriangleMesh1(mesh1NumObjects, -1);
    std::vector<int> firstTriangleMesh2(mesh2NumObjects, -1);
    for (int tri = 0; tri < mesh1NumTriangles; ++tri) {
        int obj = mesh1.triangleToObject[tri];
        if (firstTriangleMesh1[obj] == -1) {
            firstTriangleMesh1[obj] = tri;
        }
    }
    for (int tri = 0; tri < mesh2NumTriangles; ++tri) {
        int obj = mesh2.triangleToObject[tri];
        if (firstTriangleMesh2[obj] == -1) {
            firstTriangleMesh2[obj] = tri;
        }
    }

    int* d_first_triangle_mesh1 = nullptr;
    int* d_first_triangle_mesh2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh1, mesh1NumObjects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh2, mesh2NumObjects * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_first_triangle_mesh1, firstTriangleMesh1.data(), mesh1NumObjects * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first_triangle_mesh2, firstTriangleMesh2.data(), mesh2NumObjects * sizeof(int), cudaMemcpyHostToDevice));

    MeshIntersectionLaunchParams params1;
    params1.mesh1_vertices = mesh1Uploader.getVertices();
    params1.mesh1_indices = mesh1Uploader.getIndices();
    params1.mesh1_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh1_num_objects = mesh1NumObjects;
    params1.edge_starts = mesh1EdgeData.d_edge_starts;
    params1.edge_ends = mesh1EdgeData.d_edge_ends;
    params1.edge_source_object_counts = mesh1EdgeData.d_source_object_counts;
    params1.edge_source_objects = mesh1EdgeData.d_source_objects;
    params1.edge_source_object_offsets = mesh1EdgeData.d_source_object_offsets;
    params1.num_edges = mesh1NumEdges;
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
    params2.edge_starts = mesh2EdgeData.d_edge_starts;
    params2.edge_ends = mesh2EdgeData.d_edge_ends;
    params2.edge_source_object_counts = mesh2EdgeData.d_source_object_counts;
    params2.edge_source_objects = mesh2EdgeData.d_source_objects;
    params2.edge_source_object_offsets = mesh2EdgeData.d_source_object_offsets;
    params2.num_edges = mesh2NumEdges;
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

    params1.first_triangle_index_per_object = d_first_triangle_mesh1;
    params2.first_triangle_index_per_object = d_first_triangle_mesh2;

    params1.overlap_max_iterations = overlapMaxIterations;
    params1.containment_max_iterations = containmentMaxIterations;
    params1.containment_query_point_mode = containmentQueryPointMode;
    params1.profiling_enabled = enableProfilingStats ? 1 : 0;
    params1.profiling_stats = d_profiling_stats;

    params2.overlap_max_iterations = overlapMaxIterations;
    params2.containment_max_iterations = containmentMaxIterations;
    params2.containment_query_point_mode = containmentQueryPointMode;
    params2.profiling_enabled = enableProfilingStats ? 1 : 0;
    params2.profiling_stats = d_profiling_stats;

    timer.next("Query");
    std::cout << "Running Intersection Query..." << std::endl;

    QueryResults results = executeHashQuery(
        intersectionLauncher, params1, params2,
        mesh1NumEdges, mesh2NumEdges,
        mesh1NumObjects, mesh2NumObjects,
        d_hash_table, hash_table_size, queryDirection,
        &timer
    );

    std::cout << "Actual Intersection Pairs: " << results.numUnique << std::endl;
    timer.addCounter("Profile_Actual_Intersection_Pairs", static_cast<unsigned long long>(results.numUnique));
    std::cout << "Mesh1 Universe Min: [" << mesh1.grid.minBound.x << ", " << mesh1.grid.minBound.y << ", " << mesh1.grid.minBound.z << "]" << std::endl;
    std::cout << "Mesh1 Universe Max: [" << mesh1.grid.maxBound.x << ", " << mesh1.grid.maxBound.y << ", " << mesh1.grid.maxBound.z << "]" << std::endl;
    std::cout << "Mesh2 Universe Min: [" << mesh2.grid.minBound.x << ", " << mesh2.grid.minBound.y << ", " << mesh2.grid.minBound.z << "]" << std::endl;
    std::cout << "Mesh2 Universe Max: [" << mesh2.grid.maxBound.x << ", " << mesh2.grid.maxBound.y << ", " << mesh2.grid.maxBound.z << "]" << std::endl;

    if (enableProfilingStats) {
        CUDA_CHECK(cudaMemcpy(&h_profiling_stats, d_profiling_stats, sizeof(MeshIntersectionProfilingStats), cudaMemcpyDeviceToHost));

        timer.addCounter("Profile_Overlap_Trace_Calls", h_profiling_stats.overlap_trace_calls);
        timer.addCounter("Profile_Overlap_Iterations_Total", h_profiling_stats.overlap_iterations_total);
        timer.addCounter("Profile_Overlap_Hits_Total", h_profiling_stats.overlap_hits_total);
        timer.addCounter("Profile_Overlap_Max_Iterations_Per_Trace", h_profiling_stats.overlap_max_iterations_per_trace);

        timer.addCounter("Profile_Containment_Rays_Total", h_profiling_stats.containment_rays_total);
        timer.addCounter("Profile_Containment_Iterations_Total", h_profiling_stats.containment_iterations_total);
        timer.addCounter("Profile_Containment_Hits_Total", h_profiling_stats.containment_hits_total);
        timer.addCounter("Profile_Containment_Max_Iterations_Per_Ray", h_profiling_stats.containment_max_iterations_per_ray);
        timer.addCounter("Profile_Containment_Same_Hit_Suppressed", h_profiling_stats.containment_same_hit_suppressed);
        timer.addCounter("Profile_Containment_Candidate_Additions", h_profiling_stats.containment_candidate_additions);
        timer.addCounter("Profile_Containment_Candidate_Toggles", h_profiling_stats.containment_candidate_toggles);
        timer.addCounter("Profile_Containment_Candidate_Overflow", h_profiling_stats.containment_candidate_overflow);
        timer.addCounter("Profile_Containment_Targets_Total", h_profiling_stats.containment_targets_total);

        const unsigned long long overlapCalls = h_profiling_stats.overlap_trace_calls;
        const unsigned long long containmentRays = h_profiling_stats.containment_rays_total;
        const unsigned long long avgOverlapIterScaled = overlapCalls ? (h_profiling_stats.overlap_iterations_total * 1000ULL / overlapCalls) : 0ULL;
        const unsigned long long avgContainmentIterScaled = containmentRays ? (h_profiling_stats.containment_iterations_total * 1000ULL / containmentRays) : 0ULL;
        timer.addCounter("Profile_Overlap_Avg_Iterations_x1000", avgOverlapIterScaled);
        timer.addCounter("Profile_Containment_Avg_Iterations_x1000", avgContainmentIterScaled);
    }

    timer.next("Cleanup");
    if (d_profiling_stats) {
        CUDA_CHECK(cudaFree(d_profiling_stats));
    }
    CUDA_CHECK(cudaFree(d_hash_table));
    CUDA_CHECK(cudaFree(results.d_merged_results));
    CUDA_CHECK(cudaFree(d_first_triangle_mesh1));
    CUDA_CHECK(cudaFree(d_first_triangle_mesh2));
    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);

    mesh1Uploader.free();
    mesh2Uploader.free();

    timer.finish(outputJsonPath);
    return 0;
}
