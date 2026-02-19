#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "OptixContext.h"
#include "common.h"

class OptixPipelineManager {
public:
    OptixPipelineManager(OptixContext& context, const std::string& ptxPath);
    ~OptixPipelineManager();
    
    OptixPipelineManager(const OptixPipelineManager&) = delete;
    OptixPipelineManager& operator=(const OptixPipelineManager&) = delete;
    
    OptixPipeline getPipeline() const { return pipeline_; }
    OptixShaderBindingTable getSBT() const { return sbt_; }
    CUdeviceptr getLaunchParamsBuffer() const { return d_lp_; }
    
    bool isValid() const { return pipeline_ != nullptr; }
    
private:
    OptixContext& context_;
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
    OptixPipelineCompileOptions pipelineCompileOptions_;
    
    void createModule(const std::string& ptxPath);
    void createProgramGroups();
    void createPipeline();
    void createSBT();
};

