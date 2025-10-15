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
#include "common.h"
#include "dataset_loader.h"
#include "timer.h"

constexpr const char* ptxPath = "C:/Users/anton/Documents/Uni/PipRay/build/raytracing.ptx";
// constexpr const char* ptxPath = "/root/media/Spatial_Data_Management/PipRay/build/raytracing.ptx";

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
    std::string outputJsonPath = "performance_timing.json";  // Default output file
    int numberOfRuns = 1;
    
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
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--geometry <path_to_geometry_file>] [--points <path_to_point_wkt_file>] [--output <json_output_file>] [--runs <number>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --geometry <path>  Path to preprocessed geometry text file" << std::endl;
                std::cout << "  --points <path>    Path to WKT file containing POINT geometries for ray origins" << std::endl;
                std::cout << "  --output <path>    Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>    Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --help, -h         Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "OptiX multiple rays example" << std::endl;

    GeometryData geometry;
    if (!geometryFilePath.empty()) {
        geometry = loadGeometryFromFile(geometryFilePath);
    } else {
        std::cerr << "Error: Geometry file path is required. Use --geometry <path_to_geometry_file>" << std::endl;
        std::cout << "Note: You can generate a geometry file using the preprocess_dataset tool:" << std::endl;
        std::cout << "  preprocess_dataset --dataset <wkt_file> --output-geometry <geometry_file.txt>" << std::endl;
        return 1;
    }
    
    if (geometry.vertices.empty()) {
        std::cerr << "Error: Failed to load geometry." << std::endl;
        return 1;
    }

    PointData pointData = loadPointDataset(pointDatasetPath);
    if (pointData.positions.empty()) {
        std::cerr << "Error: Failed to load point dataset." << std::endl;
        return 1;
    }
    
    timer.next("Application Creation");
    
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &context));
    
    timer.next("Initialization");

    float3* d_vertices = nullptr;
    uint3*  d_indices  = nullptr;
    size_t vbytes = geometry.vertices.size()*sizeof(float3);
    size_t ibytes = geometry.indices.size()*sizeof(uint3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices),vbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices),ibytes));
    CUDA_CHECK(cudaMemcpy(d_vertices,geometry.vertices.data(),vbytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices,geometry.indices.data(),ibytes,cudaMemcpyHostToDevice));

    CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
    CUdeviceptr d_indices_ptr  = reinterpret_cast<CUdeviceptr>(d_indices);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    buildInput.triangleArray.numVertices = static_cast<unsigned int>(geometry.vertices.size());
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.indexBuffer = d_indices_ptr;
    buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(geometry.indices.size());
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.triangleArray.flags        = &triangle_input_flags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context,&accelOptions,&buildInput,1,&gasSizes));

    CUdeviceptr d_tempBuffer,d_gasOutput;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer),gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput),gasSizes.outputSizeInBytes));

    timer.next("Build Index");
    
    OptixTraversableHandle gasHandle = 0;
    OPTIX_CHECK(optixAccelBuild(context,0,&accelOptions,&buildInput,1,d_tempBuffer,gasSizes.tempSizeInBytes,d_gasOutput,gasSizes.outputSizeInBytes,&gasHandle,nullptr,0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
    
    timer.next("Module and Pipeline Setup");

    std::vector<char> ptxData = readPTX(ptxPath);
    std::cout << "PTX file loaded successfully, size: " << ptxData.size() << " bytes" << std::endl;

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
    
    std::cout << "Creating OptiX module..." << std::endl;
    OptixResult result = optixModuleCreate(context,
                                  &moduleCompileOptions,
                                  &pipelineCompileOptions,
                                  ptxData.data(),
                                  ptxData.size(),
                                  log,
                                  &sizeof_log,
                                  &module);
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed with error code: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
    
    if (sizeof_log > 1) {
        std::cout << "OptiX module log:\n" << log << std::endl;
    }
    std::cout << "OptiX module created successfully!" << std::endl;

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
    OptixProgramGroup hitPG;
    OPTIX_CHECK(optixProgramGroupCreate(context,&hitDesc,1,&pgOptions,nullptr,nullptr,&hitPG));

    std::vector<OptixProgramGroup> pgs = { raygenPG, missPG, hitPG };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(context,&pipelineCompileOptions,&linkOptions,pgs.data(),pgs.size(),nullptr,nullptr,&pipeline));

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

    RayResult h_result = {};
    CUdeviceptr d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(RayResult)));
    CUDA_CHECK(cudaMemcpy((void*)d_result, &h_result, sizeof(RayResult), cudaMemcpyHostToDevice));

    CUdeviceptr d_lp;
    CUDA_CHECK(cudaMalloc((void**)&d_lp,sizeof(LaunchParams)));

    const int numRays = static_cast<int>(pointData.numPoints);
    
    std::vector<float3> rayOrigins = pointData.positions;
    
    float3* d_ray_origins = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origins), numRays * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_ray_origins, rayOrigins.data(), numRays * sizeof(float3), cudaMemcpyHostToDevice));
    
    std::vector<RayResult> h_results(numRays);
    RayResult* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_results), numRays * sizeof(RayResult)));
    
    int* d_triangle_to_polygon = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_to_polygon), geometry.triangleToPolygon.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triangle_to_polygon, geometry.triangleToPolygon.data(), geometry.triangleToPolygon.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    timer.next("Query");
    
    if (numberOfRuns > 1) {
        std::cout << "\n=== Running " << numberOfRuns << " iterations for performance measurement ===" << std::endl;
        
        // Perform multiple runs within a single Query phase
        for (int run = 0; run < numberOfRuns; ++run) {
            std::cout << "Run " << (run + 1) << "/" << numberOfRuns << std::endl;
            
            LaunchParams lp = {};
            lp.handle = gasHandle;
            lp.ray_origins = d_ray_origins;
            lp.triangle_to_polygon = d_triangle_to_polygon;
            lp.num_rays = numRays;
            lp.result = d_results;
            
            CUDA_CHECK(cudaMemcpy((void*)d_lp, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
            
            OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRays, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else {
        LaunchParams lp = {};
        lp.handle = gasHandle;
        lp.ray_origins = d_ray_origins;
        lp.triangle_to_polygon = d_triangle_to_polygon;
        lp.num_rays = numRays;
        lp.result = d_results;
        
        CUDA_CHECK(cudaMemcpy((void*)d_lp, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
        
        std::cout << "\n=== Tracing " << numRays << " rays in a single GPU launch ===" << std::endl;
        
        OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRays, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    timer.next("Output");
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, numRays * sizeof(RayResult), cudaMemcpyDeviceToHost));

    size_t numHit = 0, numMiss = 0;
    for (int i = 0; i < numRays; ++i) {
        if (h_results[i].hit)
            ++numHit;
        else
            ++numMiss;
    }
    std::cout << "\n=== Ray Hit/Miss Summary ===" << std::endl;
    std::cout << "Total rays: " << numRays << std::endl;
    std::cout << "Hit:  " << numHit << std::endl;
    std::cout << "Miss: " << numMiss << std::endl;
    std::cout << "Hit ratio: " << (numRays > 0 ? (double)numHit / numRays * 100.0 : 0.0) << " %" << std::endl;

    if (numRays <= 100) {
        std::cout << "\n=== Ray Results ===" << std::endl;
        for (int i = 0; i < numRays; ++i) {
            std::cout << "\n=== Ray " << (i+1) << " ===" << std::endl;
            std::cout << "Ray origin: (" << rayOrigins[i].x << ", " << rayOrigins[i].y << ", " << rayOrigins[i].z << ")" << std::endl;
            
            if (h_results[i].hit) {
                // int polygonIndex = geometry.triangleToPolygon[h_results[i].triangle_index];
                std::cout << "Ray HIT the triangles!" << std::endl;
                std::cout << "  Distance: " << h_results[i].t << std::endl;
                std::cout << "  Hit point: (" << h_results[i].hit_point.x << ", " << h_results[i].hit_point.y << ", " << h_results[i].hit_point.z << ")" << std::endl;
                std::cout << "  Barycentric coordinates: (" << h_results[i].barycentrics.x << ", " << h_results[i].barycentrics.y << ")" << std::endl;
                // std::cout << "  Triangle index: " << h_results[i].triangle_index << std::endl;
                std::cout << "  Polygon index: " << h_results[i].polygon_index << std::endl;
            } else {
                std::cout << "Ray MISSED the triangles" << std::endl;
            }
        }
    } else {
        std::cout << "\n=== Ray tracing completed for " << numRays << " rays ===" << std::endl;
        std::cout << "Results are not displayed due to large number of rays." << std::endl;
    }

    std::cout << "Exporting results" << std::endl;
    std::ofstream csvFile("ray_results.csv");
    csvFile << "pointId,polygonId\n";
    for (int i = 0; i < numRays; ++i) {
        int polygonId = -1;
        if (h_results[i].hit) {
            // polygonId = geometry.triangleToPolygon[h_results[i].triangle_index];
            polygonId = h_results[i].polygon_index;
        }
        csvFile << i << "," << polygonId << "\n";
    }
    csvFile.close();

    timer.next("Cleanup");

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_results)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ray_origins)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_triangle_to_polygon)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_result)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput)));
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
    std::cout << "Number of rays processed: " << numRays << std::endl;
    
    if (numberOfRuns > 1) {
        std::cout << "Number of runs: " << numberOfRuns << std::endl;
        std::cout << "Average query time per ray per run: " << (double)timer.getPhaseDuration("Query") / (numRays * numberOfRuns) << " μs" << std::endl;
    } else {
        std::cout << "Average query time per ray: " << (double)timer.getPhaseDuration("Query") / numRays << " μs" << std::endl;
    }
    
    std::cout << "Geometry: " << geometry.vertices.size() << " vertices, " << geometry.indices.size() << " triangles" << std::endl;

    return 0;
} 