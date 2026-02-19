#pragma once

#include "common.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include <vector>

class RayLauncher {
public:
    RayLauncher(OptixPipelineManager& pipeline, GeometryUploader& geometry);
    ~RayLauncher();
    
    RayLauncher(const RayLauncher&) = delete;
    RayLauncher& operator=(const RayLauncher&) = delete;
    
    void uploadRays(const float3* rayOrigins, size_t numRays);
    void uploadRays(const std::vector<float3>& rayOrigins);
    
    void launch(OptixAccelerationStructure& as, int numRays);
    
    void runWarmup(OptixAccelerationStructure& as, int warmupCount);
    
    RayResult* getResultBuffer() const { return d_results_; }
    size_t getNumRays() const { return num_rays_; }
    
    void free();
    
    bool isUploaded() const { return d_ray_origins_ != nullptr; }
    
private:
    OptixPipelineManager& pipeline_;
    GeometryUploader& geometry_;
    float3* d_ray_origins_;
    RayResult* d_results_;
    size_t num_rays_;
    
    void freeInternal();
    void prepareLaunchParams(OptixAccelerationStructure& as, LaunchParams& lp);
};

