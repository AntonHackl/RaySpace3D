#pragma once

#include "../common.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "../geometry/GeometryUploader.h"
#include <vector>

class RayLauncher {
public:
    RayLauncher(OptixPipelineManager& pipeline, GeometryUploader& geometry);
    ~RayLauncher();
    
    // Disable copy construction and assignment
    RayLauncher(const RayLauncher&) = delete;
    RayLauncher& operator=(const RayLauncher&) = delete;
    
    // Upload ray origins to GPU
    void uploadRays(const float3* rayOrigins, size_t numRays);
    void uploadRays(const std::vector<float3>& rayOrigins);
    
    // Launch rays (single launch)
    void launch(OptixAccelerationStructure& as, int numRays);
    
    // Run warmup launches
    void runWarmup(OptixAccelerationStructure& as, int warmupCount);
    
    // Get result buffer (for processing)
    RayResult* getResultBuffer() const { return d_results_; }
    size_t getNumRays() const { return num_rays_; }
    
    // Free GPU memory
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

