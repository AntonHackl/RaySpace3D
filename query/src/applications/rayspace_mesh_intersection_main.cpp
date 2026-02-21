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
#include "../cuda/mesh_intersection.h"
#include "../cuda/mesh_overlap_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshIntersectionLauncher.h"
#include "../timer.h"
#include "../ptx_utils.h"

struct QueryResults {
    MeshOverlapResult* d_merged_results;
    int numUnique;
};

// Execute the intersection query using two-pass approach
QueryResults executeTwoPassQuery(
    MeshIntersectionLauncher& intersectionLauncher,
    MeshIntersectionLaunchParams& params1,
    MeshIntersectionLaunchParams& params2,
    int mesh1NumTriangles,
    int mesh2NumTriangles,
    bool verbose = true
) {
    // PASS 1: Count collisions
    int* d_collision_counts1 = nullptr;
    int* d_collision_counts2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_counts1, (size_t)mesh1NumTriangles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_counts2, (size_t)mesh2NumTriangles * sizeof(int)));

    params1.use_hash_table = false;
    params1.collision_counts = d_collision_counts1;
    params1.pass = 1;
    intersectionLauncher.launchMesh1ToMesh2(params1, mesh1NumTriangles);

    params2.use_hash_table = false;
    params2.collision_counts = d_collision_counts2;
    params2.pass = 1;
    intersectionLauncher.launchMesh2ToMesh1(params2, mesh2NumTriangles);

    // Scan counts
    long long* d_collision_offsets1 = nullptr;
    long long* d_collision_offsets2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_offsets1, (size_t)mesh1NumTriangles * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_collision_offsets2, (size_t)mesh2NumTriangles * sizeof(long long)));

    long long total_results1 = exclusive_scan_gpu(d_collision_counts1, d_collision_offsets1, mesh1NumTriangles);
    long long total_results2 = exclusive_scan_gpu(d_collision_counts2, d_collision_offsets2, mesh2NumTriangles);

    CUDA_CHECK(cudaFree(d_collision_counts1));
    CUDA_CHECK(cudaFree(d_collision_counts2));

    if (verbose) {
        std::cout << "Pass 1: Found " << total_results1 << " + " << total_results2
                  << " = " << (total_results1 + total_results2) << " potential intersections." << std::endl;
    }

    // PASS 2: Store results into a single merged buffer
    long long total_all = total_results1 + total_results2;
    MeshOverlapResult* d_merged_results = nullptr;
    if (total_all > 0) {
        CUDA_CHECK(cudaMalloc(&d_merged_results, (size_t)total_all * sizeof(MeshOverlapResult)));
    }

    params1.collision_offsets = d_collision_offsets1;
    params1.results = d_merged_results;
    params1.pass = 2;
    intersectionLauncher.launchMesh1ToMesh2(params1, mesh1NumTriangles);

    params2.collision_offsets = d_collision_offsets2;
    params2.results = (d_merged_results ? d_merged_results + total_results1 : nullptr);
    params2.pass = 2;
    intersectionLauncher.launchMesh2ToMesh1(params2, mesh2NumTriangles);

    CUDA_CHECK(cudaFree(d_collision_offsets1));
    CUDA_CHECK(cudaFree(d_collision_offsets2));

    int numUnique = 0;
    if (total_all > 0) {
        numUnique = (int)merge_and_deduplicate_gpu(nullptr, total_all, nullptr, 0, d_merged_results);
    }

    if (verbose) {
        std::cout << "Deduplication: Found " << numUnique << " unique object pairs." << std::endl;
    }

    return {d_merged_results, numUnique};
}

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");
    
    std::string mesh1Path = "";
    std::string mesh2Path = "";
    std::string outputJsonPath = "mesh_intersection_timing.json";
    std::string ptxPath = detectPTXPath();
    int numberOfRuns = 1;
    bool exportResults = true;
    int warmupRuns = 2;
    
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
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--mesh1 <path>] [--mesh2 <path>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mesh1 <path>         Path to first mesh dataset (geometry file)" << std::endl;
                std::cout << "  --mesh2 <path>         Path to second mesh dataset (geometry file)" << std::endl;
                std::cout << "  --output <path>        Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>        Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --warmup-runs <number> Number of warmup iterations" << std::endl;
                std::cout << "  --ptx <ptx_file>       Path to compiled PTX file (default: auto-detect)" << std::endl;
                std::cout << "  --no-export            Do not export results to CSV" << std::endl;
                std::cout << "  --help, -h             Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "Mesh-to-Mesh Intersection Join" << std::endl;
    
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
    MeshIntersectionLauncher intersectionLauncher(context, basePipeline);
    
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
    
    std::set<int> mesh1Objects(mesh1Data.triangleToObject.begin(), mesh1Data.triangleToObject.end());
    std::set<int> mesh2Objects(mesh2Data.triangleToObject.begin(), mesh2Data.triangleToObject.end());
    int mesh1NumObjects = mesh1Objects.size();
    int mesh2NumObjects = mesh2Objects.size();
    std::cout << "Mesh1 has " << mesh1NumObjects << " unique objects" << std::endl;
    std::cout << "Mesh2 has " << mesh2NumObjects << " unique objects" << std::endl;
    
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
    // Legacy params set to null
    params1.collision_counts = nullptr;
    params1.collision_offsets = nullptr;
    params1.results = nullptr; 
    params1.pass = 0; 
    // Hash params
    params1.use_hash_table = false;
    params1.hash_table = nullptr;
    params1.hash_table_size = 0;
    params1.object_tested = d_object_tested1;
    
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
    // Legacy params set to null
    params2.collision_counts = nullptr;
    params2.collision_offsets = nullptr;
    params2.results = nullptr; 
    params2.pass = 0;
    // Hash params
    params2.use_hash_table = false;
    params2.hash_table = nullptr;
    params2.hash_table_size = 0;
    params2.object_tested = d_object_tested2;
    
    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (two-pass)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            QueryResults warmupResults = executeTwoPassQuery(
                intersectionLauncher, params1, params2,
                mesh1NumTriangles, mesh2NumTriangles,
                false
            );
            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }
    
    timer.next("Query");
    
        std::cout << "\n=== Executing mesh intersection detection ===" << std::endl;
        QueryResults queryResults = executeTwoPassQuery(
            intersectionLauncher, params1, params2,
            mesh1NumTriangles, mesh2NumTriangles,
            true
        );
    
    MeshOverlapResult* d_merged_results = queryResults.d_merged_results;
    int numUnique = queryResults.numUnique;
    
    timer.next("GPU Deduplication");
    std::cout << "Deduplication: Performed via thrust sort/unique." << std::endl;
    
    timer.next("Download Results");
    
    std::vector<MeshOverlapResult> uniqueResults(numUnique);
    if (numUnique > 0) {
        CUDA_CHECK(cudaMemcpy(uniqueResults.data(), d_merged_results, 
                              numUnique * sizeof(MeshOverlapResult), 
                              cudaMemcpyDeviceToHost));
    }
    
    timer.next("Output");
    
    std::cout << "\n=== Mesh Intersection Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Mesh1 objects: " << mesh1NumObjects << std::endl;
    std::cout << "Mesh2 objects: " << mesh2NumObjects << std::endl;
    std::cout << "Unique intersecting object pairs: " << numUnique << std::endl;
    
    if (exportResults) {
        std::cout << "Exporting results to mesh_intersection_results.csv" << std::endl;
        std::ofstream csvFile("mesh_intersection_results.csv");
        csvFile << "object_id_mesh1,object_id_mesh2\n";
        for (const auto& result : uniqueResults) {
            csvFile << result.object_id_mesh1 << "," << result.object_id_mesh2 << "\n";
        }
        csvFile.close();
    }
    
    timer.next("Cleanup");
    
    if (d_merged_results) CUDA_CHECK(cudaFree(d_merged_results));
    if (d_object_tested1) CUDA_CHECK(cudaFree(d_object_tested1));
    if (d_object_tested2) CUDA_CHECK(cudaFree(d_object_tested2));
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Mesh intersection join completed successfully" << std::endl;
    
    return 0;
}
