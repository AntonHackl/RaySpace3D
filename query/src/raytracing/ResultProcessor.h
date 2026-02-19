#pragma once

#include "common.h"
#include "../cuda/result_compaction.h"
#include <vector>

class ResultProcessor {
public:
    ResultProcessor();
    ~ResultProcessor() = default;
    
    std::vector<RayResult> compactAndDownload(RayResult* d_results, int numResults);
    
    int countHits(RayResult* d_results, int numResults);
    
    // candidateIndices maps ray_id to original point index
    std::vector<int> mapToOriginalIndices(const std::vector<RayResult>& hits,
                                          const std::vector<int>& candidateIndices,
                                          size_t totalPoints);
};

