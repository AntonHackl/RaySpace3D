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
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshContainmentLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "app_cli_options.h"

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

class ContainmentCliOptions : public BenchmarkMeshPairCliOptions {
public:
    ContainmentCliOptions() : BenchmarkMeshPairCliOptions("containment_timing.json") {
        allowNoExportFlag = true;
    }

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(
            options,
            "Dataset A (container meshes)",
            "Dataset B (objects tested for containment)"
        );
        appendBenchmarkRunHelp(options);
        appendNoExportHelp(options);
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "--mesh1 <A_dataset> --mesh2 <B_dataset> [options]",
            "Containment query: finds pairs where B objects are fully contained in A objects.",
            options
        );
    }
};

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");

    ContainmentCliOptions options;
    options.ptxPath = detectPTXPath();
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }

    options.sanitizeRunCounts();

    const std::string& meshAPath = options.mesh1Path;
    const std::string& meshBPath = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const std::string& ptxPath = options.ptxPath;
    const int numberOfRuns = options.numberOfRuns;
    const int warmupRuns = options.warmupRuns;
    const bool exportResults = options.exportResults;

    std::cout << "=== Mesh Containment Query ===" << std::endl;
    std::cout << "(checks which B-objects are fully contained inside A-objects)" << std::endl;

    if (!options.hasRequiredMeshInputs()) {
        if (meshAPath.empty()) { std::cerr << "Error: --mesh1 (dataset A) required\n"; }
        if (meshBPath.empty()) { std::cerr << "Error: --mesh2 (dataset B) required\n"; }
        return 1;
    }

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
    if (!requirePrecomputedEdges(meshAData, meshAPath, "MeshA")) { return 1; }
    std::cout << "  A: " << meshAData.vertices.size() << " vertices, "
              << meshAData.indices.size() << " triangles" << std::endl;

    timer.next("Load Mesh B");
    std::cout << "Loading dataset B from: " << meshBPath << std::endl;
    GeometryData meshBData = loadGeometryFromFile(meshBPath);
    if (meshBData.vertices.empty()) { std::cerr << "Failed to load B\n"; return 1; }
    if (!requirePrecomputedEdges(meshBData, meshBPath, "MeshB")) { return 1; }
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
    EdgeMeshData aEdgeData = PrecomputedEdgeData::uploadFromGeometry(meshAData);
    EdgeMeshData bEdgeData = PrecomputedEdgeData::uploadFromGeometry(meshBData);
    int aNumEdges = aEdgeData.num_edges;
    int bNumEdges = bEdgeData.num_edges;

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

    float3* d_bFirstVertices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bFirstVertices, numBObjects * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_bFirstVertices, bFirstVertices.data(),
                          numBObjects * sizeof(float3), cudaMemcpyHostToDevice));

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
        params.src_edge_starts             = bEdgeData.d_edge_starts;
        params.src_edge_ends               = bEdgeData.d_edge_ends;
        params.src_edge_source_object_counts = bEdgeData.d_source_object_counts;
        params.src_edge_source_objects       = bEdgeData.d_source_objects;
        params.src_edge_source_object_offsets = bEdgeData.d_source_object_offsets;
        params.src_num_edges               = bNumEdges;
        params.target_handle           = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();
        params.intersection_hash_table      = d_intersectionHT;
        params.intersection_hash_table_size = intersectionHTSize;
        params.swap_ids                = 0;
        params.b_first_vertices        = d_bFirstVertices;
        params.b_num_objects           = numBObjects;
        params.containment_hash_table       = d_containmentHT;
        params.containment_hash_table_size  = containmentHTSize;

        launcher.launchEdgeCheck(params, bNumEdges);

        // Phase 1b: A edges → B
        params.src_edge_starts             = aEdgeData.d_edge_starts;
        params.src_edge_ends               = aEdgeData.d_edge_ends;
        params.src_edge_source_object_counts = aEdgeData.d_source_object_counts;
        params.src_edge_source_objects       = aEdgeData.d_source_objects;
        params.src_edge_source_object_offsets = aEdgeData.d_source_object_offsets;
        params.src_num_edges               = aNumEdges;
        params.target_handle           = bAS.getHandle();
        params.target_triangle_to_object = bUploader.getTriangleToObject();
        params.swap_ids                = 1;

        launcher.launchEdgeCheck(params, aNumEdges);

        // Phase 2: Point-in-mesh
        params.target_handle             = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();

        launcher.launchPointInMesh(params, numBObjects);

        // Compact results
        int maxOutput = 2000000;
        MeshQueryResult* d_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_results, maxOutput * sizeof(MeshQueryResult)));

        int numContained = compact_hash_table_pairs(
            d_containmentHT, containmentHTSize, d_results, maxOutput);

        if (verbose) {
            std::cout << "Containment pairs found: " << numContained << std::endl;
        }

        // Copy to host
        std::vector<MeshQueryResult> results(numContained);
        if (numContained > 0) {
            CUDA_CHECK(cudaMemcpy(results.data(), d_results,
                                  numContained * sizeof(MeshQueryResult),
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

    std::vector<MeshQueryResult> finalResults;
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
    PrecomputedEdgeData::freeEdgeData(aEdgeData);
    PrecomputedEdgeData::freeEdgeData(bEdgeData);

    timer.finish(outputJsonPath);

    std::cout << "\nContainment query completed successfully." << std::endl;
    return 0;
}
