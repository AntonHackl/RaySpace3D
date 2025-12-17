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
    
    // Disable copy construction and assignment
    MeshOverlapLauncher(const MeshOverlapLauncher&) = delete;
    MeshOverlapLauncher& operator=(const MeshOverlapLauncher&) = delete;
    
    // Launch Mesh1 edges against Mesh2 acceleration structure
    void launchMesh1ToMesh2(const MeshOverlapLaunchParams& params, int numTriangles);
    
    // Launch Mesh2 edges against Mesh1 acceleration structure
    void launchMesh2ToMesh1(const MeshOverlapLaunchParams& params, int numTriangles);
    
    bool isValid() const { return pipeline1_ != nullptr && pipeline2_ != nullptr; }
    
private:
    OptixContext& context_;
    OptixPipelineManager& basePipeline_;
    OptixModule module_;
    OptixPipeline pipeline1_;
    OptixPipeline pipeline2_;
    OptixProgramGroup raygenPG1_;
    OptixProgramGroup raygenPG2_;
    OptixProgramGroup missPG_;
    OptixProgramGroup hitPG_;
    OptixShaderBindingTable sbt1_;
    OptixShaderBindingTable sbt2_;
    CUdeviceptr d_rg1_;
    CUdeviceptr d_rg2_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_lp1_;
    CUdeviceptr d_lp2_;
    
    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};

