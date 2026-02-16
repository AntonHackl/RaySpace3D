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
#include "../cuda/mesh_containment.h"
#include "../cuda/mesh_overlap_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshContainmentLauncher.h"
#include "../timer.h"
#include "../ptx_utils.h"

// ---------------------------------------------------------------------
// Containment query
//
// Determines which B-objects are fully contained inside which A-objects.
//
// Algorithm (per pair (A_obj, B_obj)):
//   1. Phase 1 – Edge intersection:
//      Cast all B-triangle edges against A's acceleration structure AND
//      all A-triangle edges against B's acceleration structure.
//      Any (A_obj, B_obj) pair found to intersect is NOT a containment
//      pair – these are recorded in the intersection hash table.
//
//   2. Phase 2 – Point-in-mesh:
//      For every B-object, pick one vertex and cast a +Z ray against
//      A's acceleration structure.  Count surface crossings per A-object
//      (odd/even parity rule).  If the vertex is inside A_obj AND the
//      pair was NOT flagged in Phase 1 → containment.
// ---------------------------------------------------------------------

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");

    // ---------------------------------------------------------------
    // CLI
    // ---------------------------------------------------------------
    std::string meshAPath;           // dataset A (containers)
    std::string meshBPath;           // dataset B (possibly contained)
    std::string outputJsonPath = "containment_timing.json";
    std::string ptxPath        = detectPTXPath();
    int  numberOfRuns  = 1;
    bool exportResults = true;
    int  warmupRuns    = 2;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--mesh1"  && i+1 < argc) meshAPath       = argv[++i];
        else if (arg == "--mesh2"  && i+1 < argc) meshBPath       = argv[++i];
        else if (arg == "--output" && i+1 < argc) outputJsonPath  = argv[++i];
        else if (arg == "--ptx"    && i+1 < argc) ptxPath         = argv[++i];
        else if (arg == "--runs"   && i+1 < argc) numberOfRuns    = std::atoi(argv[++i]);
        else if (arg == "--warmup-runs" && i+1 < argc) warmupRuns = std::atoi(argv[++i]);
        else if (arg == "--no-export") exportResults = false;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " --mesh1 <A_dataset> --mesh2 <B_dataset> [options]\n"
                      << "Options:\n"
                      << "  --mesh1 <path>         Dataset A (container meshes)\n"
                      << "  --mesh2 <path>         Dataset B (objects to test for containment)\n"
                      << "  --output <path>        JSON timing output        [containment_timing.json]\n"
                      << "  --runs <n>             Benchmark repetitions     [1]\n"
                      << "  --warmup-runs <n>      Warmup iterations         [2]\n"
                      << "  --ptx <path>           PTX file                  [auto-detect]\n"
                      << "  --no-export            Skip CSV export\n"
                      << "  --help, -h             This message\n";
            return 0;
        }
    }

    std::cout << "=== Mesh Containment Query ===" << std::endl;
    std::cout << "(checks which B-objects are fully contained inside A-objects)" << std::endl;

    if (meshAPath.empty()) { std::cerr << "Error: --mesh1 (dataset A) required\n"; return 1; }
    if (meshBPath.empty()) { std::cerr << "Error: --mesh2 (dataset B) required\n"; return 1; }

    // ---------------------------------------------------------------
    // Load geometry
    // ---------------------------------------------------------------
    timer.next("Application Creation");

    OptixContext context;
    OptixPipelineManager basePipeline(context, ptxPath);
    MeshContainmentLauncher launcher(context, basePipeline);

    timer.next("Load Mesh A");
    std::cout << "Loading dataset A from: " << meshAPath << std::endl;
    GeometryData meshAData = loadGeometryFromFile(meshAPath);
    if (meshAData.vertices.empty()) { std::cerr << "Failed to load A\n"; return 1; }
    std::cout << "  A: " << meshAData.vertices.size() << " vertices, "
              << meshAData.indices.size() << " triangles" << std::endl;

    timer.next("Load Mesh B");
    std::cout << "Loading dataset B from: " << meshBPath << std::endl;
    GeometryData meshBData = loadGeometryFromFile(meshBPath);
    if (meshBData.vertices.empty()) { std::cerr << "Failed to load B\n"; return 1; }
    std::cout << "  B: " << meshBData.vertices.size() << " vertices, "
              << meshBData.indices.size() << " triangles" << std::endl;

    // Count unique objects
    std::set<int> aObjSet(meshAData.triangleToObject.begin(), meshAData.triangleToObject.end());
    std::set<int> bObjSet(meshBData.triangleToObject.begin(), meshBData.triangleToObject.end());
    int numAObjects = static_cast<int>(aObjSet.size());
    int numBObjects = static_cast<int>(bObjSet.size());
    std::cout << "  A objects: " << numAObjects << "   B objects: " << numBObjects << std::endl;

    // ---------------------------------------------------------------
    // Upload & build acceleration structures
    // ---------------------------------------------------------------
    timer.next("Upload Mesh A");
    GeometryUploader aUploader;
    aUploader.upload(meshAData);

    timer.next("Upload Mesh B");
    GeometryUploader bUploader;
    bUploader.upload(meshBData);

    timer.next("Build A Index");
    OptixAccelerationStructure aAS(context, aUploader);
    aAS.build();

    timer.next("Build B Index");
    OptixAccelerationStructure bAS(context, bUploader);
    bAS.build();

    // ---------------------------------------------------------------
    // Pre-compute first vertex per B-object (host side)
    // ---------------------------------------------------------------
    timer.next("Prepare Kernel Parameters");

    int aNumTriangles = static_cast<int>(aUploader.getNumIndices());
    int bNumTriangles = static_cast<int>(bUploader.getNumIndices());

    std::vector<float3> bFirstVertices(numBObjects);
    {
        std::vector<bool> seen(numBObjects, false);
        for (int tri = 0; tri < bNumTriangles; ++tri) {
            int obj = meshBData.triangleToObject[tri];
            if (!seen[obj]) {
                uint3  idx = meshBData.indices[tri];
                bFirstVertices[obj] = meshBData.vertices[idx.x];
                seen[obj] = true;
            }
        }
    }

    // Upload to GPU
    float3* d_bFirstVertices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bFirstVertices, numBObjects * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_bFirstVertices, bFirstVertices.data(),
                          numBObjects * sizeof(float3), cudaMemcpyHostToDevice));

    // Allocate hash tables
    int intersectionHTSize  = 16777216;  // 16 M slots ≈ 128 MB
    int containmentHTSize   = 16777216;

    unsigned long long* d_intersectionHT = nullptr;
    unsigned long long* d_containmentHT  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_intersectionHT, (size_t)intersectionHTSize * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_containmentHT,  (size_t)containmentHTSize  * sizeof(unsigned long long)));

    // ------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------
    timer.next("Warmup");

    auto runOnce = [&](bool verbose) {
        CUDA_CHECK(cudaMemset(d_intersectionHT, 0xFF,
                              (size_t)intersectionHTSize * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_containmentHT,  0xFF,
                              (size_t)containmentHTSize  * sizeof(unsigned long long)));

        MeshContainmentLaunchParams params{};

        // Phase 1a: B edges → A
        params.src_vertices            = bUploader.getVertices();
        params.src_indices             = bUploader.getIndices();
        params.src_triangle_to_object  = bUploader.getTriangleToObject();
        params.src_num_triangles       = bNumTriangles;
        params.target_handle           = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();
        params.intersection_hash_table      = d_intersectionHT;
        params.intersection_hash_table_size = intersectionHTSize;
        params.swap_ids                = 0;
        params.b_first_vertices        = d_bFirstVertices;
        params.b_num_objects           = numBObjects;
        params.containment_hash_table       = d_containmentHT;
        params.containment_hash_table_size  = containmentHTSize;

        launcher.launchEdgeCheck(params, bNumTriangles);

        // Phase 1b: A edges → B
        params.src_vertices            = aUploader.getVertices();
        params.src_indices             = aUploader.getIndices();
        params.src_triangle_to_object  = aUploader.getTriangleToObject();
        params.src_num_triangles       = aNumTriangles;
        params.target_handle           = bAS.getHandle();
        params.target_triangle_to_object = bUploader.getTriangleToObject();
        params.swap_ids                = 1;

        launcher.launchEdgeCheck(params, aNumTriangles);

        // Phase 2: Point-in-mesh
        params.target_handle             = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();

        launcher.launchPointInMesh(params, numBObjects);

        // Compact results
        int maxOutput = 2000000;
        MeshOverlapResult* d_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_results, maxOutput * sizeof(MeshOverlapResult)));

        int numContained = compact_hash_table(
            d_containmentHT, containmentHTSize, d_results, maxOutput);

        if (verbose) {
            std::cout << "Containment pairs found: " << numContained << std::endl;
        }

        // Copy to host
        std::vector<MeshOverlapResult> results(numContained);
        if (numContained > 0) {
            CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                                  numContained * sizeof(MeshOverlapResult),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(d_results));

        return results;
    };

    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations..." << std::endl;
        for (int w = 0; w < warmupRuns; ++w) runOnce(false);
    }

    // ---------------------------------------------------------------
    // Timed run(s)
    // ---------------------------------------------------------------
    timer.next("Query");

    std::cout << "\n=== Executing containment query ===" << std::endl;

    std::vector<MeshOverlapResult> finalResults;
    for (int run = 0; run < numberOfRuns; ++run) {
        finalResults = runOnce(run == numberOfRuns - 1);
    }
    int numContained = static_cast<int>(finalResults.size());

    timer.next("Download Results");

    // ---------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------
    timer.next("Output");

    std::cout << "\n=== Containment Query Summary ===" << std::endl;
    std::cout << "A triangles: " << aNumTriangles << "  objects: " << numAObjects << std::endl;
    std::cout << "B triangles: " << bNumTriangles << "  objects: " << numBObjects << std::endl;
    std::cout << "Containment pairs (B in A): " << numContained << std::endl;

    if (exportResults) {
        std::string csvFile = "mesh_containment_results.csv";
        std::cout << "Exporting results to " << csvFile << std::endl;
        std::ofstream csv(csvFile);
        csv << "a_object_id,b_object_id\n";
        for (const auto& r : finalResults) {
            csv << r.object_id_mesh1 << "," << r.object_id_mesh2 << "\n";
        }
    }

    // ---------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------
    timer.next("Cleanup");

    CUDA_CHECK(cudaFree(d_intersectionHT));
    CUDA_CHECK(cudaFree(d_containmentHT));
    CUDA_CHECK(cudaFree(d_bFirstVertices));

    timer.finish(outputJsonPath);

    std::cout << "\nContainment query completed successfully." << std::endl;
    return 0;
}
