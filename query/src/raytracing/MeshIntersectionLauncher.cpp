#include "MeshIntersectionLauncher.h"
#include "../optix/OptixPipeline.h"
#include "../ptx_utils.h"
#include <iostream>

MeshIntersectionLauncher::MeshIntersectionLauncher(OptixContext& context, OptixPipelineManager& basePipeline)
    : context_(context), basePipeline_(basePipeline),
    module_(nullptr), pipeline_(nullptr),
    raygenOverlapPG_(nullptr), raygenContainmentPG_(nullptr),
    missPG_(nullptr), hitPG_(nullptr),
    d_rg_overlap_(0), d_rg_containment_(0),
    d_ms_(0), d_hg_(0), d_lp_(0) {
    
    createModule();
    createProgramGroups();
    createPipelines();
    createSBT();
}

MeshIntersectionLauncher::~MeshIntersectionLauncher() {
    freeInternal();
}

void MeshIntersectionLauncher::createModule() {
    // Load mesh_intersection.ptx (compiled from mesh_intersection.cu)
    std::string ptxPath = detectPTXPath();
    size_t pos = ptxPath.find("raytracing.ptx");
    if (pos != std::string::npos) {
        ptxPath.replace(pos, std::string("raytracing.ptx").size(), "mesh_intersection.ptx");
    } else {
        // Fallback: try to find mesh_intersection.ptx in the same directory
        size_t lastSlash = ptxPath.find_last_of("\\/");
        if (lastSlash != std::string::npos) {
            ptxPath = ptxPath.substr(0, lastSlash + 1) + "mesh_intersection.ptx";
        } else {
            ptxPath = "mesh_intersection.ptx";
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
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_intersection_params";
    
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
        std::cerr << "OptiX module creation failed for mesh intersection: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
}

void MeshIntersectionLauncher::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    
    OptixProgramGroupDesc rgOverlap = {};
    rgOverlap.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgOverlap.raygen.module = module_;
    rgOverlap.raygen.entryFunctionName = "__raygen__mesh_overlap";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &rgOverlap, 1, &pgOptions, nullptr, nullptr, &raygenOverlapPG_));

    OptixProgramGroupDesc rgContain = {};
    rgContain.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgContain.raygen.module = module_;
    rgContain.raygen.entryFunctionName = "__raygen__mesh_containment";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &rgContain, 1, &pgOptions, nullptr, nullptr, &raygenContainmentPG_));
    
    // Miss program group
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module_;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &missDesc, 1, &pgOptions, nullptr, nullptr, &missPG_));
    
    // Hit program group
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module_;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitDesc.hitgroup.moduleAH = module_;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &hitDesc, 1, &pgOptions, nullptr, nullptr, &hitPG_));
}

void MeshIntersectionLauncher::createPipelines() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_intersection_params";
    
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;
    
    std::vector<OptixProgramGroup> pgs2 = {
        raygenOverlapPG_,
        raygenContainmentPG_,
        missPG_,
        hitPG_
    };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    pgs2.data(), pgs2.size(), nullptr, nullptr, &pipeline_));
}

void MeshIntersectionLauncher::createSBT() {
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    
    RaygenRecord rgOverlap = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenOverlapPG_, &rgOverlap));

    RaygenRecord rgContain = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenContainmentPG_, &rgContain));
    
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &msRecord));
    
    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_, &hgRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_overlap_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_containment_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_), sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_), sizeof(HitRecord)));
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_overlap_), &rgOverlap, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_containment_), &rgContain, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_), &msRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_), &hgRecord, sizeof(HitRecord), cudaMemcpyHostToDevice));
    
    sbt_ = {};
    sbt_.raygenRecord = d_rg_overlap_;
    sbt_.missRecordBase = d_ms_;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = d_hg_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt_.hitgroupRecordCount = 1;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lp_), sizeof(MeshIntersectionLaunchParams)));
}

void MeshIntersectionLauncher::launchInternal(const MeshIntersectionLaunchParams& params, int launchSize, CUdeviceptr raygenRecord) {
    CUDA_CHECK(cudaMemcpy((void*)d_lp_, &params, sizeof(MeshIntersectionLaunchParams), cudaMemcpyHostToDevice));
    sbt_.raygenRecord = raygenRecord;
    OPTIX_CHECK(optixLaunch(pipeline_, 0, d_lp_, sizeof(MeshIntersectionLaunchParams), &sbt_, launchSize, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshIntersectionLauncher::launchOverlapMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 0;
    launchInternal(launchParams, launchSize, d_rg_overlap_);
}

void MeshIntersectionLauncher::launchOverlapMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 1;
    launchInternal(launchParams, launchSize, d_rg_overlap_);
}

void MeshIntersectionLauncher::launchContainmentMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 0;
    launchInternal(launchParams, launchSize, d_rg_containment_);
}

void MeshIntersectionLauncher::launchContainmentMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 1;
    launchInternal(launchParams, launchSize, d_rg_containment_);
}

void MeshIntersectionLauncher::launchMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    launchOverlapMesh1ToMesh2(params, launchSize);
}

void MeshIntersectionLauncher::launchMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    launchOverlapMesh2ToMesh1(params, launchSize);
}

void MeshIntersectionLauncher::freeInternal() {
    if (d_lp_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp_)));
    if (d_hg_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_)));
    if (d_ms_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_)));
    if (d_rg_containment_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_containment_)));
    if (d_rg_overlap_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_overlap_)));
    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (hitPG_) optixProgramGroupDestroy(hitPG_);
    if (missPG_) optixProgramGroupDestroy(missPG_);
    if (raygenContainmentPG_) optixProgramGroupDestroy(raygenContainmentPG_);
    if (raygenOverlapPG_) optixProgramGroupDestroy(raygenOverlapPG_);
    if (module_) optixModuleDestroy(module_);
}
