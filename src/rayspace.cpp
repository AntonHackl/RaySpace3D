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

// Helper function to split comma-separated file paths
static std::vector<std::string> splitPaths(const std::string& input) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        size_t start = item.find_first_not_of(" \t");
        size_t end = item.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            result.push_back(item.substr(start, end - start + 1));
        } else if (start != std::string::npos) {
            result.push_back(item.substr(start));
        }
    }
    return result;
}

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
    std::string ptxPath = detectPTXPath();                   // Auto-detected PTX path (overridable via CLI)
    int numberOfRuns = 1;
    bool exportResults = true; // set to false with --no-export
    int warmupRuns = 2; // number of warmup launches before timed Query
    
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
                std::cout << "Usage: " << argv[0] << " [--geometry <path(s)>] [--points <path(s)>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --geometry <path(s)>  Path(s) to preprocessed geometry text file(s). Use commas to separate multiple files." << std::endl;
                std::cout << "  --points <path(s)>    Path(s) to WKT file(s) containing POINT geometries. Use commas to separate multiple files." << std::endl;
                std::cout << "  --output <path>       Path to JSON file for performance timing output" << std::endl;
                std::cout << "  --runs <number>       Number of times to run the query (for performance testing)" << std::endl;
                std::cout << "  --ptx <ptx_file>      Path to compiled PTX file (default: ./raytracing.ptx)" << std::endl;
                std::cout << "  --help, -h            Show this help message" << std::endl;
                std::cout << "\nList behavior:" << std::endl;
                std::cout << "  - If one argument is a list and the other is single, the single item is used for all entries in the list." << std::endl;
                std::cout << "  - If both are lists, they must have the same length." << std::endl;
                return 0;
            }
        }
    }
    
    std::cout << "OptiX multiple rays example" << std::endl;

    // Parse file lists
    std::vector<std::string> geometryFiles = splitPaths(geometryFilePath);
    std::vector<std::string> pointFiles = splitPaths(pointDatasetPath);
    
    if (geometryFiles.empty()) {
        std::cerr << "Error: Geometry file path is required. Use --geometry <path_to_geometry_file>" << std::endl;
        std::cout << "Note: You can generate a geometry file using the preprocess_dataset tool:" << std::endl;
        std::cout << "  preprocess_dataset --dataset <wkt_file> --output-geometry <geometry_file.txt>" << std::endl;
        return 1;
    }
    
    if (pointFiles.empty()) {
        std::cerr << "Error: Points file path is required. Use --points <path_to_point_file>" << std::endl;
        return 1;
    }
    
    // Validate list lengths
    size_t numTasks = std::max(geometryFiles.size(), pointFiles.size());
    if (geometryFiles.size() > 1 && pointFiles.size() > 1 && geometryFiles.size() != pointFiles.size()) {
        std::cerr << "Error: When both --geometry and --points are lists, they must have the same length." << std::endl;
        std::cerr << "  Geometry files: " << geometryFiles.size() << std::endl;
        std::cerr << "  Point files: " << pointFiles.size() << std::endl;
        return 1;
    }
    
    std::cout << "Processing " << numTasks << " task(s)" << std::endl;
    if (geometryFiles.size() == 1 && numTasks > 1) {
        std::cout << "  Using single geometry file for all tasks: " << geometryFiles[0] << std::endl;
    }
    if (pointFiles.size() == 1 && numTasks > 1) {
        std::cout << "  Using single points file for all tasks: " << pointFiles[0] << std::endl;
    }
    
    timer.next("Application Creation");
    
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &context));
    
    timer.next("Module and Pipeline Setup");
 
    std::vector<char> ptxData = readPTX(ptxPath.c_str());
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
    hitDesc.hitgroup.moduleAH = module;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
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

    CUdeviceptr d_lp;
    CUDA_CHECK(cudaMalloc((void**)&d_lp,sizeof(LaunchParams)));
    
    // Caching variables for geometry and points
    std::string cachedGeometryPath = "";
    GeometryData cachedGeometry;
    float3* d_vertices = nullptr;
    uint3* d_indices = nullptr;
    int* d_triangle_to_object = nullptr;
    CUdeviceptr d_gasOutput = 0;
    OptixTraversableHandle gasHandle = 0;
    
    std::string cachedPointsPath = "";
    PointData cachedPointData;
    float3* d_ray_origins = nullptr;
    RayResult* d_results = nullptr;
    RayResult* d_compact_results = nullptr;
    int* d_hit_counter = nullptr;
    
    // Process each task
    for (size_t taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
        std::string currentGeomPath = geometryFiles[geometryFiles.size() == 1 ? 0 : taskIdx];
        std::string currentPointsPath = pointFiles[pointFiles.size() == 1 ? 0 : taskIdx];
        
        std::cout << "\n=== Task " << (taskIdx + 1) << "/" << numTasks << " ===" << std::endl;
        std::cout << "Geometry: " << currentGeomPath << std::endl;
        std::cout << "Points: " << currentPointsPath << std::endl;
        
        // Load/cache geometry (read into memory only)
        bool geometryChanged = (currentGeomPath != cachedGeometryPath);
        if (geometryChanged) {
            timer.next("Load Geometry");
            std::cout << "Loading new geometry..." << std::endl;
            
            cachedGeometry = loadGeometryFromFile(currentGeomPath);
            if (cachedGeometry.vertices.empty()) {
                std::cerr << "Error: Failed to load geometry from " << currentGeomPath << std::endl;
                continue;
            }
            cachedGeometryPath = currentGeomPath;
            // numRays already declared after points are loaded
        } else {
            std::cout << "Using cached geometry" << std::endl;
        }
        
        // Load/cache points (read into memory only)
        bool pointsChanged = (currentPointsPath != cachedPointsPath);
        if (pointsChanged) {
            timer.next("Load Points");
            std::cout << "Loading new points..." << std::endl;
            
            cachedPointData = loadPointDataset(currentPointsPath);
            if (cachedPointData.positions.empty()) {
                std::cerr << "Error: Failed to load points from " << currentPointsPath << std::endl;
                continue;
            }
            cachedPointsPath = currentPointsPath;
            std::cout << "Points loaded: " << cachedPointData.numPoints << std::endl;
        } else {
            std::cout << "Using cached points" << std::endl;
        }
        
        const int numRays = static_cast<int>(cachedPointData.numPoints);
        
        // Upload points to GPU
        if (pointsChanged) {
            timer.next("Upload Points");
            std::cout << "Uploading points to GPU..." << std::endl;
            
            // Free old points GPU memory
            if (d_ray_origins) CUDA_CHECK(cudaFree(d_ray_origins));
            if (d_results) CUDA_CHECK(cudaFree(d_results));
            if (d_compact_results) CUDA_CHECK(cudaFree(d_compact_results));
            if (d_hit_counter) CUDA_CHECK(cudaFree(d_hit_counter));
            
            // Ensure any previous GPU work is complete before timing upload
            CUDA_CHECK(cudaDeviceSynchronize());
            
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origins), numRays * sizeof(float3)));
            CUDA_CHECK(cudaMemcpy(d_ray_origins, cachedPointData.positions.data(), numRays * sizeof(float3), cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_results), numRays * sizeof(RayResult)));
            // Touch allocated device memory to ensure pages are resident and avoid
            // first-launch page faults. Also synchronize to ensure uploads complete.
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_results), 0, numRays * sizeof(RayResult)));
            
            // Allocate compact output buffers (worst case: all rays hit)
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compact_results), numRays * sizeof(RayResult)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_counter), sizeof(int)));
            
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "Points uploaded to GPU" << std::endl;
        }
        
        // Upload geometry to GPU and build index
        if (geometryChanged) {
            timer.next("Upload Geometry");
            std::cout << "Uploading geometry to GPU..." << std::endl;
            
            // Free old geometry GPU memory
            if (d_vertices) CUDA_CHECK(cudaFree(d_vertices));
            if (d_indices) CUDA_CHECK(cudaFree(d_indices));
            if (d_triangle_to_object) CUDA_CHECK(cudaFree(d_triangle_to_object));
            if (d_gasOutput) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput)));
            
            size_t vbytes = cachedGeometry.vertices.size() * sizeof(float3);
            size_t ibytes = cachedGeometry.indices.size() * sizeof(uint3);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vbytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), ibytes));
            CUDA_CHECK(cudaMemcpy(d_vertices, cachedGeometry.vertices.data(), vbytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_indices, cachedGeometry.indices.data(), ibytes, cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_to_object), cachedGeometry.triangleToObject.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_triangle_to_object, cachedGeometry.triangleToObject.data(), cachedGeometry.triangleToObject.size() * sizeof(int), cudaMemcpyHostToDevice));
            
            std::cout << "Geometry uploaded to GPU" << std::endl;
            
            // Build acceleration structure
            timer.next("Build Index");
            std::cout << "Building acceleration structure..." << std::endl;
            CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(d_vertices);
            CUdeviceptr d_indices_ptr = reinterpret_cast<CUdeviceptr>(d_indices);
            
            OptixBuildInput buildInput = {};
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            buildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
            buildInput.triangleArray.numVertices = static_cast<unsigned int>(cachedGeometry.vertices.size());
            buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
            buildInput.triangleArray.indexBuffer = d_indices_ptr;
            buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(cachedGeometry.indices.size());
            buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
            unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
            buildInput.triangleArray.flags = &triangle_input_flags;
            buildInput.triangleArray.numSbtRecords = 1;
            
            OptixAccelBuildOptions accelOptions = {};
            accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
            
            OptixAccelBufferSizes gasSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasSizes));
            
            CUdeviceptr d_tempBuffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasSizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput), gasSizes.outputSizeInBytes));
            
            OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &buildInput, 1, d_tempBuffer, gasSizes.tempSizeInBytes, d_gasOutput, gasSizes.outputSizeInBytes, &gasHandle, nullptr, 0));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
            // Ensure the acceleration structure build is completed on the device
            // and visible to subsequent launches.
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "Acceleration structure built" << std::endl;
        }
        
        timer.next("Warmup");
        const int warmupRuns = 2;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            // Reset hit counter for warmup
            CUDA_CHECK(cudaMemset(d_hit_counter, 0, sizeof(int)));
            
            LaunchParams lp = {};
            lp.handle = gasHandle;
            lp.ray_origins = d_ray_origins;
            lp.indices = d_indices;
            lp.triangle_to_object = d_triangle_to_object;
            lp.num_rays = numRays;
            lp.result = d_results;
            lp.compact_result = d_compact_results;
            lp.hit_counter = d_hit_counter;
            
            CUDA_CHECK(cudaMemcpy((void*)d_lp, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
            
            OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRays, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        timer.next("Query");

        if (numberOfRuns > 1) {
            std::cout << "\n=== Running " << numberOfRuns << " iterations for performance measurement ===" << std::endl;
            
            // Perform multiple runs within a single Query phase
            for (int run = 0; run < numberOfRuns; ++run) {
                std::cout << "Run " << (run + 1) << "/" << numberOfRuns << std::endl;
                
                // Reset hit counter for each run
                CUDA_CHECK(cudaMemset(d_hit_counter, 0, sizeof(int)));
                
                LaunchParams lp = {};
                lp.handle = gasHandle;
                lp.ray_origins = d_ray_origins;
                lp.indices = d_indices;
                lp.triangle_to_object = d_triangle_to_object;
                lp.num_rays = numRays;
                lp.result = d_results;
                lp.compact_result = d_compact_results;
                lp.hit_counter = d_hit_counter;
                
                CUDA_CHECK(cudaMemcpy((void*)d_lp, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
                
                OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRays, 1, 1));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        } else {
            // Reset hit counter
            CUDA_CHECK(cudaMemset(d_hit_counter, 0, sizeof(int)));
            
            LaunchParams lp = {};
            lp.handle = gasHandle;
            lp.ray_origins = d_ray_origins;
            lp.indices = d_indices;
            lp.triangle_to_object = d_triangle_to_object;
            lp.num_rays = numRays;
            lp.result = d_results;
            lp.compact_result = d_compact_results;
            lp.hit_counter = d_hit_counter;
            
            CUDA_CHECK(cudaMemcpy((void*)d_lp, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
            
            std::cout << "\n=== Tracing " << numRays << " rays in a single GPU launch ===" << std::endl;
            
            OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp, sizeof(LaunchParams), &sbt, numRays, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        timer.next("Download Results");
        // Download hit count from GPU
        int numHit = 0;
        CUDA_CHECK(cudaMemcpy(&numHit, d_hit_counter, sizeof(int), cudaMemcpyDeviceToHost));
        size_t numMiss = numRays - numHit;
        
        // Download only compacted hits
        std::vector<RayResult> h_hits;
        if (numHit > 0) {
            h_hits.resize(numHit);
            CUDA_CHECK(cudaMemcpy(h_hits.data(), d_compact_results, numHit * sizeof(RayResult), cudaMemcpyDeviceToHost));
        }
        timer.next("Output");
        std::cout << "\n=== Point-in-Polygon Summary ===" << std::endl;
        std::cout << "Total rays: " << numRays << std::endl;
        std::cout << "Points INSIDE polygons:  " << numHit << std::endl;
        std::cout << "Points OUTSIDE polygons: " << numMiss << std::endl;
        std::cout << "Inside ratio: " << (numRays > 0 ? (double)numHit / numRays * 100.0 : 0.0) << " %" << std::endl;
        std::cout << "Memory saved: " << ((numRays - numHit) * sizeof(RayResult)) / (1024.0 * 1024.0) << " MB by not downloading misses" << std::endl;

        if (numRays <= 100) {
            std::cout << "\n=== Ray Results ===" << std::endl;
            // Create a lookup for hits
            std::vector<bool> isHit(numRays, false);
            std::vector<RayResult> hitLookup(numRays);
            for (const auto& hit : h_hits) {
                isHit[hit.ray_id] = true;
                hitLookup[hit.ray_id] = hit;
            }
            
            for (int i = 0; i < numRays; ++i) {
                std::cout << "\n=== Ray " << (i+1) << " ===" << std::endl;
                std::cout << "Ray origin: (" << cachedPointData.positions[i].x << ", " << cachedPointData.positions[i].y << ", " << cachedPointData.positions[i].z << ")" << std::endl;

                if (isHit[i]) {
                    const auto& result = hitLookup[i];
                    std::cout << "Point is INSIDE a polygon (ray entering back face)" << std::endl;
                    std::cout << "  Polygon index: " << result.polygon_index << std::endl;
                } else {
                    std::cout << "Point is OUTSIDE all polygons (no hit or ray entering front face)" << std::endl;
                }
            }
        } else {
            std::cout << "\n=== Ray tracing completed for " << numRays << " rays ===" << std::endl;
            std::cout << "Results are not displayed due to large number of rays." << std::endl;
        }

        if (exportResults) {
            std::cout << "Exporting results (hits only)" << std::endl;
            std::ofstream csvFile("ray_results.csv");
            csvFile << "pointId,polygonId\n";
            for (const auto& hit : h_hits) {
                csvFile << hit.ray_id << "," << hit.polygon_index << "\n";
            }
            csvFile.close();
            std::cout << "Exported " << numHit << " hit points (" << numMiss << " misses excluded)" << std::endl;
        } else {
            std::cout << "Skipping export of results (disabled by flag)" << std::endl;
        }
    } // End of task loop

    timer.next("Cleanup");
    
    // Free cached GPU memory
    if (d_results) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_results)));
    if (d_compact_results) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_compact_results)));
    if (d_hit_counter) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hit_counter)));
    if (d_ray_origins) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ray_origins)));
    if (d_triangle_to_object) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_triangle_to_object)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg)));
    if (d_gasOutput) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput)));
    if (d_vertices) CUDA_CHECK(cudaFree(d_vertices));
    if (d_indices) CUDA_CHECK(cudaFree(d_indices));

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygenPG);
    optixProgramGroupDestroy(missPG);
    optixProgramGroupDestroy(hitPG);
    optixModuleDestroy(module);
    optixDeviceContextDestroy(context);
    
    timer.finish(outputJsonPath);
    
    std::cout << std::endl;
    std::cout << "Total tasks processed: " << numTasks << std::endl;
    
    if (numberOfRuns > 1) {
        std::cout << "Number of runs per task: " << numberOfRuns << std::endl;
    }
    
    if (!cachedGeometry.vertices.empty()) {
        std::cout << "Final geometry: " << cachedGeometry.vertices.size() << " vertices, " << cachedGeometry.indices.size() << " triangles" << std::endl;
    }

    return 0;
} 