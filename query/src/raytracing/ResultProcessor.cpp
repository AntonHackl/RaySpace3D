#include "ResultProcessor.h"
#include "../optix/OptixHelpers.h"
#include <vector>

ResultProcessor::ResultProcessor() {
}

std::vector<RayResult> ResultProcessor::compactAndDownload(RayResult* d_results, int numResults) {
    std::vector<RayResult> hits;
    
    if (numResults <= 0 || d_results == nullptr) {
        return hits;
    }
    
    int numHits = countHits(d_results, numResults);
    
    if (numHits > 0) {
        hits.resize(numHits);
        
        RayResult* d_compact = nullptr;
        CUDA_CHECK(cudaMalloc(&d_compact, numHits * sizeof(RayResult)));
        
        int actual_hits = 0;
        compact_hits_gpu(d_results, d_compact, numResults, &actual_hits);
        
        CUDA_CHECK(cudaMemcpy(hits.data(), d_compact, actual_hits * sizeof(RayResult), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_compact));
        
        if (actual_hits != numHits) {
            hits.resize(actual_hits);
        }
    }
    
    return hits;
}

int ResultProcessor::countHits(RayResult* d_results, int numResults) {
    if (numResults <= 0 || d_results == nullptr) {
        return 0;
    }
    return count_hits_gpu(d_results, numResults);
}

std::vector<int> ResultProcessor::mapToOriginalIndices(const std::vector<RayResult>& hits,
                                                        const std::vector<int>& candidateIndices,
                                                        size_t totalPoints) {
    std::vector<int> finalResults(totalPoints, -1); // -1 means outside
    
    for (const auto& hit : hits) {
        // hit.ray_id is the index in candidateIndices array
        if (hit.ray_id >= 0 && static_cast<size_t>(hit.ray_id) < candidateIndices.size()) {
            int originalIdx = candidateIndices[hit.ray_id];
            if (originalIdx >= 0 && static_cast<size_t>(originalIdx) < totalPoints) {
                finalResults[originalIdx] = hit.polygon_index;
            }
        }
    }
    
    return finalResults;
}

