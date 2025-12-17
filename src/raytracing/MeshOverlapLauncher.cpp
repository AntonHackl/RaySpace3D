#include "MeshOverlapLauncher.h"
#include "../optix/OptixPipeline.h"
#include "../ptx_utils.h"
#include <iostream>

MeshOverlapLauncher::MeshOverlapLauncher(OptixContext& context, OptixPipelineManager& basePipeline)
    : context_(context), basePipeline_(basePipeline),
      module_(nullptr), pipeline1_(nullptr), pipeline2_(nullptr),
      raygenPG1_(nullptr), raygenPG2_(nullptr), missPG_(nullptr), hitPG_(nullptr),
      d_rg1_(0), d_rg2_(0), d_ms_(0), d_hg_(0), d_lp1_(0), d_lp2_(0) {
    
    // Reuse the same PTX file (mesh_overlap.cu should be compiled into the same PTX)
    // For now, we'll use the existing raytracing.ptx which should include mesh_overlap functions
    // In a full implementation, mesh_overlap.cu would be compiled separately or included in raytracing.ptx
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
    // mesh_overlap.ptx contains the raygen programs and shaders
    std::string ptxPath = detectPTXPath();
    // Replace "raytracing.ptx" with "mesh_overlap.ptx"
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
    
    // Raygen program group 1: Mesh1 to Mesh2
    OptixProgramGroupDesc raygenDesc1 = {};
    raygenDesc1.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc1.raygen.module = module_;
    raygenDesc1.raygen.entryFunctionName = "__raygen__mesh1_to_mesh2";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &raygenDesc1, 1, &pgOptions, nullptr, nullptr, &raygenPG1_));
    
    // Raygen program group 2: Mesh2 to Mesh1
    OptixProgramGroupDesc raygenDesc2 = {};
    raygenDesc2.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc2.raygen.module = module_;
    raygenDesc2.raygen.entryFunctionName = "__raygen__mesh2_to_mesh1";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &raygenDesc2, 1, &pgOptions, nullptr, nullptr, &raygenPG2_));
    
    // Reuse miss and hit groups from base pipeline (they're the same)
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
    
    // Pipeline 1: Mesh1 to Mesh2
    std::vector<OptixProgramGroup> pgs1 = { raygenPG1_, missPG_, hitPG_ };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    pgs1.data(), pgs1.size(), nullptr, nullptr, &pipeline1_));
    
    // Pipeline 2: Mesh2 to Mesh1
    std::vector<OptixProgramGroup> pgs2 = { raygenPG2_, missPG_, hitPG_ };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    pgs2.data(), pgs2.size(), nullptr, nullptr, &pipeline2_));
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
    
    RaygenRecord rgRecord1 = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG1_, &rgRecord1));
    
    RaygenRecord rgRecord2 = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG2_, &rgRecord2));
    
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &msRecord));
    
    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_, &hgRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg1_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg2_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_), sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_), sizeof(HitRecord)));
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg1_), &rgRecord1, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg2_), &rgRecord2, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_), &msRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_), &hgRecord, sizeof(HitRecord), cudaMemcpyHostToDevice));
    
    sbt1_ = {};
    sbt1_.raygenRecord = d_rg1_;
    sbt1_.missRecordBase = d_ms_;
    sbt1_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt1_.missRecordCount = 1;
    sbt1_.hitgroupRecordBase = d_hg_;
    sbt1_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt1_.hitgroupRecordCount = 1;
    
    sbt2_ = {};
    sbt2_.raygenRecord = d_rg2_;
    sbt2_.missRecordBase = d_ms_;
    sbt2_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt2_.missRecordCount = 1;
    sbt2_.hitgroupRecordBase = d_hg_;
    sbt2_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt2_.hitgroupRecordCount = 1;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lp1_), sizeof(MeshOverlapLaunchParams)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lp2_), sizeof(MeshOverlapLaunchParams)));
}

void MeshOverlapLauncher::launchMesh1ToMesh2(const MeshOverlapLaunchParams& params, int numTriangles) {
    CUDA_CHECK(cudaMemcpy((void*)d_lp1_, &params, sizeof(MeshOverlapLaunchParams), cudaMemcpyHostToDevice));
    
    OPTIX_CHECK(optixLaunch(pipeline1_, 0, d_lp1_, 
                            sizeof(MeshOverlapLaunchParams), &sbt1_, numTriangles, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshOverlapLauncher::launchMesh2ToMesh1(const MeshOverlapLaunchParams& params, int numTriangles) {
    CUDA_CHECK(cudaMemcpy((void*)d_lp2_, &params, sizeof(MeshOverlapLaunchParams), cudaMemcpyHostToDevice));
    
    OPTIX_CHECK(optixLaunch(pipeline2_, 0, d_lp2_, 
                            sizeof(MeshOverlapLaunchParams), &sbt2_, numTriangles, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshOverlapLauncher::freeInternal() {
    if (d_lp2_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp2_)));
    if (d_lp1_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp1_)));
    if (d_hg_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_)));
    if (d_ms_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_)));
    if (d_rg2_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg2_)));
    if (d_rg1_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg1_)));
    if (pipeline2_) optixPipelineDestroy(pipeline2_);
    if (pipeline1_) optixPipelineDestroy(pipeline1_);
    if (hitPG_) optixProgramGroupDestroy(hitPG_);
    if (missPG_) optixProgramGroupDestroy(missPG_);
    if (raygenPG2_) optixProgramGroupDestroy(raygenPG2_);
    if (raygenPG1_) optixProgramGroupDestroy(raygenPG1_);
    if (module_) optixModuleDestroy(module_);
}

