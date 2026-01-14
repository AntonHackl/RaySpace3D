#pragma once

#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../cuda/mesh_intersection.h"
#include "../optix/OptixHelpers.h"
#include <optix.h>

class MeshIntersectionLauncher {
public:
    MeshIntersectionLauncher(OptixContext& context, OptixPipelineManager& basePipeline);
    ~MeshIntersectionLauncher();
    
    // Disable copy construction and assignment
    MeshIntersectionLauncher(const MeshIntersectionLauncher&) = delete;
    MeshIntersectionLauncher& operator=(const MeshIntersectionLauncher&) = delete;
    
    // Launch Mesh1 edges against Mesh2 acceleration structure
    void launchMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int numTriangles);
    
    // Launch Mesh2 edges against Mesh1 acceleration structure
    void launchMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int numTriangles);
    
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
