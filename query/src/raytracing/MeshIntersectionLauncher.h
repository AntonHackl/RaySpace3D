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
    
    MeshIntersectionLauncher(const MeshIntersectionLauncher&) = delete;
    MeshIntersectionLauncher& operator=(const MeshIntersectionLauncher&) = delete;
    
    void launchOverlapMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchOverlapMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchContainmentMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchContainmentMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize);
    
    bool isValid() const { return pipeline_ != nullptr; }
    
private:
    OptixContext& context_;
    OptixPipelineManager& basePipeline_;
    OptixModule module_;
    OptixPipeline pipeline_;
    OptixProgramGroup raygenOverlap12PG_;
    OptixProgramGroup raygenOverlap21PG_;
    OptixProgramGroup raygenContainment12PG_;
    OptixProgramGroup raygenContainment21PG_;
    OptixProgramGroup missPG_;
    OptixProgramGroup hitPG_;
    OptixShaderBindingTable sbt_;
    CUdeviceptr d_rg_overlap12_;
    CUdeviceptr d_rg_overlap21_;
    CUdeviceptr d_rg_containment12_;
    CUdeviceptr d_rg_containment21_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_lp_;

    void launchInternal(const MeshIntersectionLaunchParams& params, int launchSize, CUdeviceptr raygenRecord);
    
    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};
