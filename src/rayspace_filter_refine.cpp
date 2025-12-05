// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif
#include "common.h"
#include "dataset/common/Geometry.h"
#include "dataset/runtime/GeometryIO.h"
#include "dataset/runtime/PointIO.h"
#include "timer.h"
#include "result_compaction.h"
#include "ptx_utils.h"

// Bounding box structure
struct BoundingBox {
    float3 min;
    float3 max;
    
    BoundingBox() {
        min.x = min.y = min.z = std::numeric_limits<float>::max();
        max.x = max.y = max.z = -std::numeric_limits<float>::max();
    }
    
    void expand(const float3& point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }
};

static std::vector<char> readPTX(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

#define OPTIX_CHECK(call) do { OptixResult res = call; if(res!=OPTIX_SUCCESS){ std::cerr << "OptiX error " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(EXIT_FAILURE);} } while(0)
#define CUDA_CHECK(call) do { cudaError_t err = call; if(err!=cudaSuccess){ std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(EXIT_FAILURE);} } while(0)

int main(int argc, char* argv[])
{
    PerformanceTimer timer;
    timer.start("Data Reading");
    
    std::string geometryFilePath = "";
    std::string pointDatasetPath = "";
    std::string outputJsonPath = "performance_timing_filter_refine.json";
    std::string ptxPath = detectPTXPath();  // Auto-detected PTX path (overridable via CLI)
    int numberOfRuns = 1;
    bool exportResults = true;
    int warmupRuns = 2;
    
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
    if (pointData.positions.empty()) {
        std::cerr << "Error: Failed to load points from " << pointDatasetPath << std::endl;
        return 1;
    }
    std::cout << "Points loaded: " << pointData.numPoints << std::endl;
    
    timer.next("Application Creation");
    
    // Initialize OptiX
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &context));
    
    timer.next("Module and Pipeline Setup");
    
    // Load PTX and create module
    std::vector<char> ptxData = readPTX(ptxPath.c_str());

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur        = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 3;
    pipelineCompileOptions.numAttributeValues    = 2;
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixModule module = nullptr;
    char log[8192];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(context, &moduleCompileOptions, &pipelineCompileOptions,
                                   ptxData.data(), ptxData.size(), log, &sizeof_log, &module));

    // Create program groups
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";
    OptixProgramGroup raygenPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&raygenDesc,1,&pgOptions,nullptr,nullptr,&raygenPG));

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OptixProgramGroup missPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&missDesc,1,&pgOptions,nullptr,nullptr,&missPG));

    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitDesc.hitgroup.moduleAH = module;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OptixProgramGroup hitPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&hitDesc,1,&pgOptions,nullptr,nullptr,&hitPG));

    std::vector<OptixProgramGroup> pgs = { raygenPG, missPG, hitPG };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(context,&pipelineCompileOptions,&linkOptions,pgs.data(),pgs.size(),nullptr,nullptr,&pipeline));

    // Create SBT
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; RayGenData data; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord    { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; HitGroupData data; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord     { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; HitGroupData data; };

    RaygenRecord rgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG,&rgRecord));
    rgRecord.data.origin = {0.0f,0.0f,0.0f};
    rgRecord.data.direction = {0.0f,0.0f,1.0f};

    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG,&msRecord));

    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG,&hgRecord));

    CUdeviceptr d_rg,d_ms,d_hg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg),sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms),sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg),sizeof(HitRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg),&rgRecord,sizeof(RaygenRecord),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms),&msRecord,sizeof(MissRecord),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg),&hgRecord,sizeof(HitRecord),cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord            = d_rg;
    sbt.missRecordBase          = d_ms;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = 1;
    sbt.hitgroupRecordBase      = d_hg;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt.hitgroupRecordCount     = 1;

    CUdeviceptr d_lp;
    CUDA_CHECK(cudaMalloc((void**)&d_lp,sizeof(LaunchParams)));
    
    // Upload all points to GPU before filter phase
    timer.next("Upload Points");
    float3* d_bbox_ray_origins = nullptr;
    RayResult* d_bbox_results = nullptr;
    
    const int numBBoxRays = static_cast<int>(pointData.numPoints);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_ray_origins), numBBoxRays * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_bbox_ray_origins, pointData.positions.data(), numBBoxRays * sizeof(float3), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_results), numBBoxRays * sizeof(RayResult)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_bbox_results), 0, numBBoxRays * sizeof(RayResult)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // PHASE 1: FILTER - Compute bounding box of query geometry
    timer.next("Filter");
    std::cout << "\n=== PHASE 1: FILTER (Bounding Box via Ray Tracing) ===" << std::endl;

    BoundingBox queryBBox;
    for (const auto& vertex : geometry.vertices) {
        queryBBox.expand(vertex);
    }
    
    std::cout << "Query bounding box computed:" << std::endl;
    std::cout << "  Min: (" << queryBBox.min.x << ", " << queryBBox.min.y << ", " << queryBBox.min.z << ")" << std::endl;
    std::cout << "  Max: (" << queryBBox.max.x << ", " << queryBBox.max.y << ", " << queryBBox.max.z << ")" << std::endl;

    // Create bounding box geometry (8 vertices, 12 triangles forming a box)
    std::vector<float3> bboxVertices;
    bboxVertices.reserve(8);
    float3 v0; v0.x = queryBBox.min.x; v0.y = queryBBox.min.y; v0.z = queryBBox.min.z; bboxVertices.push_back(v0); // 0
    float3 v1; v1.x = queryBBox.max.x; v1.y = queryBBox.min.y; v1.z = queryBBox.min.z; bboxVertices.push_back(v1); // 1
    float3 v2; v2.x = queryBBox.max.x; v2.y = queryBBox.max.y; v2.z = queryBBox.min.z; bboxVertices.push_back(v2); // 2
    float3 v3; v3.x = queryBBox.min.x; v3.y = queryBBox.max.y; v3.z = queryBBox.min.z; bboxVertices.push_back(v3); // 3
    float3 v4; v4.x = queryBBox.min.x; v4.y = queryBBox.min.y; v4.z = queryBBox.max.z; bboxVertices.push_back(v4); // 4
    float3 v5; v5.x = queryBBox.max.x; v5.y = queryBBox.min.y; v5.z = queryBBox.max.z; bboxVertices.push_back(v5); // 5
    float3 v6; v6.x = queryBBox.max.x; v6.y = queryBBox.max.y; v6.z = queryBBox.max.z; bboxVertices.push_back(v6); // 6
    float3 v7; v7.x = queryBBox.min.x; v7.y = queryBBox.max.y; v7.z = queryBBox.max.z; bboxVertices.push_back(v7); // 7
    
    // 12 triangles forming the box (with outward-facing winding order)
    std::vector<uint3> bboxIndices;
    bboxIndices.reserve(12);
    // Bottom face (z = min)
    uint3 t0; t0.x = 0; t0.y = 2; t0.z = 1; bboxIndices.push_back(t0);
    uint3 t1; t1.x = 0; t1.y = 3; t1.z = 2; bboxIndices.push_back(t1);
    // Top face (z = max)
    uint3 t2; t2.x = 4; t2.y = 5; t2.z = 6; bboxIndices.push_back(t2);
    uint3 t3; t3.x = 4; t3.y = 6; t3.z = 7; bboxIndices.push_back(t3);
    // Front face (y = min)
    uint3 t4; t4.x = 0; t4.y = 1; t4.z = 5; bboxIndices.push_back(t4);
    uint3 t5; t5.x = 0; t5.y = 5; t5.z = 4; bboxIndices.push_back(t5);
    // Back face (y = max)
    uint3 t6; t6.x = 3; t6.y = 6; t6.z = 2; bboxIndices.push_back(t6);
    uint3 t7; t7.x = 3; t7.y = 7; t7.z = 6; bboxIndices.push_back(t7);
    // Left face (x = min)
    uint3 t8; t8.x = 0; t8.y = 4; t8.z = 7; bboxIndices.push_back(t8);
    uint3 t9; t9.x = 0; t9.y = 7; t9.z = 3; bboxIndices.push_back(t9);
    // Right face (x = max)
    uint3 t10; t10.x = 1; t10.y = 2; t10.z = 6; bboxIndices.push_back(t10);
    uint3 t11; t11.x = 1; t11.y = 6; t11.z = 5; bboxIndices.push_back(t11);
    
    // Triangle to object mapping (all belong to object 0)
    std::vector<int> bboxTriangleToObject(bboxIndices.size(), 0);
    
    // Upload bbox geometry to GPU
    float3* d_bbox_vertices = nullptr;
    uint3* d_bbox_indices = nullptr;
    int* d_bbox_tri_to_obj = nullptr;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_vertices), bboxVertices.size() * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_indices), bboxIndices.size() * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_tri_to_obj), bboxTriangleToObject.size() * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_bbox_vertices, bboxVertices.data(), bboxVertices.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bbox_indices, bboxIndices.data(), bboxIndices.size() * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bbox_tri_to_obj, bboxTriangleToObject.data(), bboxTriangleToObject.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Build acceleration structure for bbox
    CUdeviceptr d_bbox_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_bbox_vertices);
    CUdeviceptr d_bbox_indices_ptr = reinterpret_cast<CUdeviceptr>(d_bbox_indices);
    
    OptixBuildInput bboxBuildInput = {};
    bboxBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bboxBuildInput.triangleArray.vertexBuffers = &d_bbox_vertices_ptr;
    bboxBuildInput.triangleArray.numVertices = static_cast<unsigned int>(bboxVertices.size());
    bboxBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bboxBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    bboxBuildInput.triangleArray.indexBuffer = d_bbox_indices_ptr;
    bboxBuildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(bboxIndices.size());
    bboxBuildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bboxBuildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    unsigned int bbox_triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    bboxBuildInput.triangleArray.flags = &bbox_triangle_input_flags;
    bboxBuildInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions bboxAccelOptions = {};
    bboxAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    bboxAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes bboxGasSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &bboxAccelOptions, &bboxBuildInput, 1, &bboxGasSizes));
    
    CUdeviceptr d_bbox_tempBuffer;
    CUdeviceptr d_bbox_gasOutput;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_tempBuffer), bboxGasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bbox_gasOutput), bboxGasSizes.outputSizeInBytes));
    
    OptixTraversableHandle bboxGasHandle;
    OPTIX_CHECK(optixAccelBuild(context, 0, &bboxAccelOptions, &bboxBuildInput, 1, 
                                 d_bbox_tempBuffer, bboxGasSizes.tempSizeInBytes, 
                                 d_bbox_gasOutput, bboxGasSizes.outputSizeInBytes, 
                                 &bboxGasHandle, nullptr, 0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_tempBuffer)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Launch bbox filter
    LaunchParams bboxLp = {};
    bboxLp.handle = bboxGasHandle;
    bboxLp.ray_origins = d_bbox_ray_origins;
    bboxLp.indices = d_bbox_indices;
    bboxLp.triangle_to_object = d_bbox_tri_to_obj;
    bboxLp.num_rays = numBBoxRays;
    bboxLp.result = d_bbox_results;
    
    CUDA_CHECK(cudaMemcpy((void*)d_lp, &bboxLp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
    
    std::cout << "Launching bounding box filter with " << numBBoxRays << " rays..." << std::endl;
    OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numBBoxRays, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Compact bbox filter results on GPU - only get hits
    // Count hits on GPU
    int numBBoxHits = count_hits_gpu(d_bbox_results, numBBoxRays);
    
    // Download only bbox hits
    std::vector<RayResult> bboxHits;
    if (numBBoxHits > 0) {
        bboxHits.resize(numBBoxHits);
        // Allocate temporary GPU buffer for compacted results
        RayResult* d_bbox_compact;
        CUDA_CHECK(cudaMalloc(&d_bbox_compact, numBBoxHits * sizeof(RayResult)));
        
        // Compact on GPU
        int actual_bbox_hits = 0;
        compact_hits_gpu(d_bbox_results, d_bbox_compact, numBBoxRays, &actual_bbox_hits);
        
        // Download compacted results
        CUDA_CHECK(cudaMemcpy(bboxHits.data(), d_bbox_compact, actual_bbox_hits * sizeof(RayResult), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_bbox_compact));
        
        numBBoxHits = actual_bbox_hits;
    }
    
    // Extract candidate indices from hits
    std::vector<int> candidateIndices;
    candidateIndices.reserve(bboxHits.size());
    for (const auto& hit : bboxHits) {
        candidateIndices.push_back(hit.ray_id);
    }
    
    size_t totalPoints = pointData.numPoints;
    size_t candidateCount = candidateIndices.size();
    size_t filteredOut = totalPoints - candidateCount;
    
    std::cout << "Bounding box filter results:" << std::endl;
    std::cout << "  Total points:           " << totalPoints << std::endl;
    std::cout << "  Candidates (in bbox):   " << candidateCount << " (" 
              << (totalPoints > 0 ? (candidateCount * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "  Filtered out:           " << filteredOut << " (" 
              << (totalPoints > 0 ? (filteredOut * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    
    // Cleanup bbox resources (but keep OptiX context, pipeline, and SBT for reuse)
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_results)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_ray_origins)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_tri_to_obj)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_indices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_bbox_gasOutput)));
    
    if (candidateCount == 0) {
        std::cout << "\nNo points inside bounding box. No refinement needed." << std::endl;
        
        if (exportResults) {
            std::cout << "Exporting results" << std::endl;
            std::ofstream csvFile("ray_results.csv");
            csvFile << "pointId,polygonId\n";
            for (size_t i = 0; i < totalPoints; ++i) {
                csvFile << i << ",-1\n";
            }
            csvFile.close();
        }
        
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg)));
        optixPipelineDestroy(pipeline);
        optixProgramGroupDestroy(raygenPG);
        optixProgramGroupDestroy(missPG);
        optixProgramGroupDestroy(hitPG);
        optixModuleDestroy(module);
        optixDeviceContextDestroy(context);
        
        timer.finish(outputJsonPath);
        return 0;
    }

    // PHASE 2: REFINE - Exact raytracing for candidate points
    timer.next("Upload Points");
    std::cout << "\n=== PHASE 2: REFINE (Exact Raytracing) ===" << std::endl;
    std::cout << "Processing " << candidateCount << " candidate points with raytracing..." << std::endl;
    
    // Prepare candidate points for raytracing
    std::vector<float3> candidatePoints;
    candidatePoints.reserve(candidateCount);
    for (int idx : candidateIndices) {
        candidatePoints.push_back(pointData.positions[idx]);
    }
    
    float3* d_refine_ray_origins = nullptr;
    RayResult* d_refine_results = nullptr;
    
    const int numRefineRays = static_cast<int>(candidateCount);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_refine_ray_origins), numRefineRays * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_refine_ray_origins, candidatePoints.data(), numRefineRays * sizeof(float3), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_refine_results), numRefineRays * sizeof(RayResult)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_refine_results), 0, numRefineRays * sizeof(RayResult)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Upload query geometry to GPU
    timer.next("Upload Geometry");
    float3* d_vertices = nullptr;
    uint3* d_indices = nullptr;
    int* d_triangle_to_object = nullptr;
    
    size_t vbytes = geometry.vertices.size() * sizeof(float3);
    size_t ibytes = geometry.indices.size() * sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), ibytes));
    CUDA_CHECK(cudaMemcpy(d_vertices, geometry.vertices.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, geometry.indices.data(), ibytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_to_object), geometry.triangleToObject.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triangle_to_object, geometry.triangleToObject.data(), geometry.triangleToObject.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Build acceleration structure for query geometry
    timer.next("Build Index");
    CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
    CUdeviceptr d_indices_ptr = reinterpret_cast<CUdeviceptr>(d_indices);
    
    OptixBuildInput queryBuildInput = {};
    queryBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    queryBuildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    queryBuildInput.triangleArray.numVertices = static_cast<unsigned int>(geometry.vertices.size());
    queryBuildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    queryBuildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    queryBuildInput.triangleArray.indexBuffer = d_indices_ptr;
    queryBuildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(geometry.indices.size());
    queryBuildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    queryBuildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    unsigned int query_triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    queryBuildInput.triangleArray.flags = &query_triangle_input_flags;
    queryBuildInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions queryAccelOptions = {};
    queryAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    queryAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes queryGasSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &queryAccelOptions, &queryBuildInput, 1, &queryGasSizes));
    
    CUdeviceptr d_query_tempBuffer;
    CUdeviceptr d_query_gasOutput;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_tempBuffer), queryGasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_gasOutput), queryGasSizes.outputSizeInBytes));
    
    OptixTraversableHandle queryGasHandle;
    OPTIX_CHECK(optixAccelBuild(context, 0, &queryAccelOptions, &queryBuildInput, 1, 
                                 d_query_tempBuffer, queryGasSizes.tempSizeInBytes, 
                                 d_query_gasOutput, queryGasSizes.outputSizeInBytes, 
                                 &queryGasHandle, nullptr, 0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_query_tempBuffer)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Warmup runs
    timer.next("Warmup");
    for (int warmup = 0; warmup < warmupRuns; ++warmup) {
        LaunchParams refineLp = {};
        refineLp.handle = queryGasHandle;
        refineLp.ray_origins = d_refine_ray_origins;
        refineLp.indices = d_indices;
        refineLp.triangle_to_object = d_triangle_to_object;
        refineLp.num_rays = numRefineRays;
        refineLp.result = d_refine_results;
        
        CUDA_CHECK(cudaMemcpy((void*)d_lp, &refineLp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
        
        OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRefineRays, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    timer.next("Query");

    if (numberOfRuns > 1) {
        std::cout << "\n=== Running " << numberOfRuns << " iterations for performance measurement ===" << std::endl;
        
        for (int run = 0; run < numberOfRuns; ++run) {
            std::cout << "Run " << (run + 1) << "/" << numberOfRuns << std::endl;
            
            LaunchParams refineLp = {};
            refineLp.handle = queryGasHandle;
            refineLp.ray_origins = d_refine_ray_origins;
            refineLp.indices = d_indices;
            refineLp.triangle_to_object = d_triangle_to_object;
            refineLp.num_rays = numRefineRays;
            refineLp.result = d_refine_results;
            
            CUDA_CHECK(cudaMemcpy((void*)d_lp, &refineLp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
            
            OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRefineRays, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else {
        LaunchParams refineLp = {};
        refineLp.handle = queryGasHandle;
        refineLp.ray_origins = d_refine_ray_origins;
        refineLp.indices = d_indices;
        refineLp.triangle_to_object = d_triangle_to_object;
        refineLp.num_rays = numRefineRays;
        refineLp.result = d_refine_results;
        
        CUDA_CHECK(cudaMemcpy((void*)d_lp, &refineLp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
        
        std::cout << "\n=== Tracing " << numRefineRays << " candidate rays ===" << std::endl;
        
        OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRefineRays, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    timer.next("Download Results");
    // Count hits on GPU
    int numHit = count_hits_gpu(d_refine_results, numRefineRays);
    // Download only hits
    std::vector<RayResult> h_refine_hits;
    if (numHit > 0) {
        h_refine_hits.resize(numHit);
        // Allocate temporary GPU buffer for compacted results
        RayResult* d_refine_compact;
        CUDA_CHECK(cudaMalloc(&d_refine_compact, numHit * sizeof(RayResult)));
        
        // Compact on GPU
        int actual_refine_hits = 0;
        compact_hits_gpu(d_refine_results, d_refine_compact, numRefineRays, &actual_refine_hits);
        
        // Download compacted results
        CUDA_CHECK(cudaMemcpy(h_refine_hits.data(), d_refine_compact, actual_refine_hits * sizeof(RayResult), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_refine_compact));
        
        numHit = actual_refine_hits;
    }
    
    timer.next("Output");
    // Map results back to original point indices
    std::vector<int> finalResults(totalPoints, -1); // -1 means outside
    
    for (const auto& hit : h_refine_hits) {
        // hit.ray_id is the index in candidateIndices array
        int originalIdx = candidateIndices[hit.ray_id];
        finalResults[originalIdx] = hit.polygon_index;
    }
    
    size_t numMiss = totalPoints - numHit;
    
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Total points:            " << totalPoints << std::endl;
    std::cout << "Filtered by bbox:        " << filteredOut << " (" 
              << (totalPoints > 0 ? (filteredOut * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Candidates tested:       " << candidateCount << " (" 
              << (totalPoints > 0 ? (candidateCount * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Points INSIDE polygons:  " << numHit << " (" 
              << (totalPoints > 0 ? (numHit * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Points OUTSIDE polygons: " << numMiss << " (" 
              << (totalPoints > 0 ? (numMiss * 100.0 / totalPoints) : 0.0) << "%)" << std::endl;
    std::cout << "Memory saved: " << ((candidateCount - numHit) * sizeof(RayResult)) / (1024.0 * 1024.0) << " MB by not downloading candidate misses" << std::endl;

    if (candidateCount <= 100) {
        std::cout << "\n=== Ray Results (Candidates Only) ===" << std::endl;
        // Create lookup for hits
        std::vector<bool> candidateIsHit(candidateCount, false);
        std::vector<RayResult> candidateHitLookup(candidateCount);
        for (const auto& hit : h_refine_hits) {
            candidateIsHit[hit.ray_id] = true;
            candidateHitLookup[hit.ray_id] = hit;
        }
        
        for (size_t i = 0; i < candidateCount; ++i) {
            int originalIdx = candidateIndices[i];
            std::cout << "\n=== Candidate " << (i+1) << " (Original Point " << originalIdx << ") ===" << std::endl;
            std::cout << "Ray origin: (" << candidatePoints[i].x << ", " << candidatePoints[i].y << ", " << candidatePoints[i].z << ")" << std::endl;
            
            if (candidateIsHit[i]) {
                const auto& result = candidateHitLookup[i];
                std::cout << "Point is INSIDE a polygon (ray entering back face)" << std::endl;
                std::cout << "  Polygon index: " << result.polygon_index << std::endl;
            } else {
                std::cout << "Point is OUTSIDE all polygons (no hit or ray entering front face)" << std::endl;
            }
        }
    }

    if (exportResults) {
        std::cout << "Exporting results (hits only)" << std::endl;
        std::ofstream csvFile("ray_results.csv");
        csvFile << "pointId,polygonId\n";
        for (const auto& hit : h_refine_hits) {
            int originalIdx = candidateIndices[hit.ray_id];
            csvFile << originalIdx << "," << hit.polygon_index << "\n";
        }
        csvFile.close();
        std::cout << "Exported " << numHit << " hit points (" << numMiss << " misses excluded)" << std::endl;
    } else {
        std::cout << "Skipping export of results (disabled by flag)" << std::endl;
    }

    timer.next("Cleanup");
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_refine_results)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_refine_ray_origins)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_triangle_to_object)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_query_gasOutput)));
    CUDA_CHECK(cudaFree(d_vertices));
    CUDA_CHECK(cudaFree(d_indices));

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygenPG);
    optixProgramGroupDestroy(missPG);
    optixProgramGroupDestroy(hitPG);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Filter-Refine processing complete!" << std::endl;
    if (numberOfRuns > 1) {
        std::cout << "Number of runs: " << numberOfRuns << std::endl;
    }

    return 0;
}
