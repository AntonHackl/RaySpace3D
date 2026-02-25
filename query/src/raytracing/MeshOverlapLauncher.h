#pragma once

#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../cuda/mesh_overlap.h"
#include "../optix/OptixHelpers.h"
#include <optix.h>

class MeshOverlapLauncher {
public:
    MeshOverlapLauncher(OptixContext& context, OptixPipelineManager& basePipeline);
    ~MeshOverlapLauncher();
    
    MeshOverlapLauncher(const MeshOverlapLauncher&) = delete;
    MeshOverlapLauncher& operator=(const MeshOverlapLauncher&) = delete;
    
    void launchMesh1ToMesh2(const MeshOverlapLaunchParams& params, int numTriangles);
    
    void launchMesh2ToMesh1(const MeshOverlapLaunchParams& params, int numTriangles);
    
    bool isValid() const { return pipeline_ != nullptr; }
    
private:
    OptixContext& context_;
    OptixPipelineManager& basePipeline_;
    OptixModule module_;
    OptixPipeline pipeline_;
    OptixProgramGroup raygenPG_;
    OptixProgramGroup missPG_;
    OptixProgramGroup hitPG_;
    OptixShaderBindingTable sbt_;
    CUdeviceptr d_rg_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_lp_;

    void launchInternal(const MeshOverlapLaunchParams& params, int numTriangles, int swapResultIds);
    
    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};

