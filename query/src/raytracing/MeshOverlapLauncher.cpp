#include "MeshOverlapLauncher.h"
#include "../optix/OptixPipeline.h"
#include "../ptx_utils.h"
#include <iostream>

MeshOverlapLauncher::MeshOverlapLauncher(OptixContext& context, OptixPipelineManager& basePipeline)
    : context_(context), basePipeline_(basePipeline),
    module_(nullptr), pipeline_(nullptr),
    raygenPG_(nullptr), missPG_(nullptr), hitPG_(nullptr),
    d_rg_(0), d_ms_(0), d_hg_(0), d_lp_(0) {
    
    createModule();
    createProgramGroups();
    createPipelines();
    createSBT();
}

MeshOverlapLauncher::~MeshOverlapLauncher() {
    freeInternal();
}

void MeshOverlapLauncher::createModule() {
    // Load mesh_overlap.ptx (compiled from mesh_overlap.cu)
    std::string ptxPath = detectPTXPath();
    size_t pos = ptxPath.find("raytracing.ptx");
    if (pos != std::string::npos) {
        ptxPath.replace(pos, std::string("raytracing.ptx").size(), "mesh_overlap.ptx");
    } else {
        // Fallback: try to find mesh_overlap.ptx in the same directory
        size_t lastSlash = ptxPath.find_last_of("\\/");
        if (lastSlash != std::string::npos) {
            ptxPath = ptxPath.substr(0, lastSlash + 1) + "mesh_overlap.ptx";
        } else {
            ptxPath = "mesh_overlap.ptx";
        }
    }
    std::vector<char> ptxData = readPTX(ptxPath.c_str());
    
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_overlap_params";
    
    char log[8192];
    size_t sizeof_log = sizeof(log);
    
    OptixResult result = optixModuleCreate(context_.getContext(),
                                          &moduleCompileOptions,
                                          &pipelineCompileOptions,
                                          ptxData.data(),
                                          ptxData.size(),
                                          log,
                                          &sizeof_log,
                                          &module_);
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed for mesh overlap: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
}

void MeshOverlapLauncher::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module_;
    raygenDesc.raygen.entryFunctionName = "__raygen__mesh_overlap";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &raygenDesc, 1, &pgOptions, nullptr, nullptr, &raygenPG_));
    
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module_;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &missDesc, 1, &pgOptions, nullptr, nullptr, &missPG_));
    
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module_;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitDesc.hitgroup.moduleAH = module_;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &hitDesc, 1, &pgOptions, nullptr, nullptr, &hitPG_));
}

void MeshOverlapLauncher::createPipelines() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_overlap_params";
    
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;
    
    std::vector<OptixProgramGroup> pgs = { raygenPG_, missPG_, hitPG_ };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    pgs.data(), pgs.size(), nullptr, nullptr, &pipeline_));
}

void MeshOverlapLauncher::createSBT() {
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    
    RaygenRecord rgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &rgRecord));
    
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &msRecord));
    
    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_, &hgRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_), sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_), sizeof(HitRecord)));
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_), &rgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_), &msRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_), &hgRecord, sizeof(HitRecord), cudaMemcpyHostToDevice));
    
    sbt_ = {};
    sbt_.raygenRecord = d_rg_;
    sbt_.missRecordBase = d_ms_;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = d_hg_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt_.hitgroupRecordCount = 1;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lp_), sizeof(MeshOverlapLaunchParams)));
}

void MeshOverlapLauncher::launchMesh1ToMesh2(const MeshOverlapLaunchParams& params, int numTriangles) {
    launchInternal(params, numTriangles, 0);
}

void MeshOverlapLauncher::launchMesh2ToMesh1(const MeshOverlapLaunchParams& params, int numTriangles) {
    launchInternal(params, numTriangles, 1);
}

void MeshOverlapLauncher::launchInternal(const MeshOverlapLaunchParams& params, int numTriangles, int swapResultIds) {
    MeshOverlapLaunchParams launchParams = params;
    launchParams.swap_result_ids = swapResultIds;
    CUDA_CHECK(cudaMemcpy((void*)d_lp_, &launchParams, sizeof(MeshOverlapLaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline_, 0, d_lp_,
                            sizeof(MeshOverlapLaunchParams), &sbt_, numTriangles, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshOverlapLauncher::freeInternal() {
    if (d_lp_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp_)));
    if (d_hg_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_)));
    if (d_ms_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_)));
    if (d_rg_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_)));
    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (hitPG_) optixProgramGroupDestroy(hitPG_);
    if (missPG_) optixProgramGroupDestroy(missPG_);
    if (raygenPG_) optixProgramGroupDestroy(raygenPG_);
    if (module_) optixModuleDestroy(module_);
}

