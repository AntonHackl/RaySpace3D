#pragma once

#include "../common.h"
#include "../cuda/result_compaction.h"
#include <vector>

class ResultProcessor {
public:
    ResultProcessor();
    ~ResultProcessor() = default;
    
    // Compact results on GPU and download hits only
    // Returns vector of hit results
    std::vector<RayResult> compactAndDownload(RayResult* d_results, int numResults);
    
    // Count hits on GPU
    int countHits(RayResult* d_results, int numResults);
    
    // Map results back to original point indices (for filter-refine)
    // candidateIndices maps ray_id to original point index
    std::vector<int> mapToOriginalIndices(const std::vector<RayResult>& hits,
                                          const std::vector<int>& candidateIndices,
                                          size_t totalPoints);
};

