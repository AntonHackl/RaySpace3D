#include "RayLauncher.h"
#include "../optix/OptixHelpers.h"
#include <iostream>

RayLauncher::RayLauncher(OptixPipelineManager& pipeline, GeometryUploader& geometry)
    : pipeline_(pipeline), geometry_(geometry),
      d_ray_origins_(nullptr), d_results_(nullptr), num_rays_(0) {
}

RayLauncher::~RayLauncher() {
    freeInternal();
}

void RayLauncher::uploadRays(const float3* rayOrigins, size_t numRays) {
    freeInternal();
    
    num_rays_ = numRays;
    if (num_rays_ == 0) {
        return;
    }
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ray_origins_), num_rays_ * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_ray_origins_, rayOrigins, num_rays_ * sizeof(float3), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_results_), num_rays_ * sizeof(RayResult)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_results_), 0, num_rays_ * sizeof(RayResult)));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void RayLauncher::uploadRays(const std::vector<float3>& rayOrigins) {
    uploadRays(rayOrigins.data(), rayOrigins.size());
}

void RayLauncher::prepareLaunchParams(OptixAccelerationStructure& as, LaunchParams& lp) {
    lp.handle = as.getHandle();
    lp.ray_origins = d_ray_origins_;
    lp.indices = geometry_.getIndices();
    lp.triangle_to_object = geometry_.getTriangleToObject();
    lp.num_rays = static_cast<int>(num_rays_);
    lp.result = d_results_;
}

void RayLauncher::launch(OptixAccelerationStructure& as, int numRays) {
    if (!isUploaded() || numRays <= 0) {
        return;
    }
    
    LaunchParams lp = {};
    prepareLaunchParams(as, lp);
    
    CUDA_CHECK(cudaMemcpy((void*)pipeline_.getLaunchParamsBuffer(), &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));
    
    OptixShaderBindingTable sbt = pipeline_.getSBT();
    OPTIX_CHECK(optixLaunch(pipeline_.getPipeline(), 0, pipeline_.getLaunchParamsBuffer(), 
                            sizeof(LaunchParams), &sbt, numRays, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void RayLauncher::runWarmup(OptixAccelerationStructure& as, int warmupCount) {
    for (int i = 0; i < warmupCount; ++i) {
        launch(as, static_cast<int>(num_rays_));
    }
}

void RayLauncher::free() {
    freeInternal();
}

void RayLauncher::freeInternal() {
    if (d_results_) {
        CUDA_CHECK(cudaFree(d_results_));
        d_results_ = nullptr;
    }
    if (d_ray_origins_) {
        CUDA_CHECK(cudaFree(d_ray_origins_));
        d_ray_origins_ = nullptr;
    }
    num_rays_ = 0;
}

