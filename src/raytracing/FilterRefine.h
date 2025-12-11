#pragma once

#include "../optix/OptixContext.h"
#include "../optix/OptixPipeline.h"
#include "../optix/OptixAccelerationStructure.h"
#include "../geometry/GeometryUploader.h"
#include "../geometry/BoundingBox.h"
#include "RayLauncher.h"
#include "ResultProcessor.h"
#include "../dataset/common/Geometry.h"
#include "../dataset/runtime/PointIO.h"
#include <vector>

// Result structure for filter-refine operations
struct FilterRefineResult {
    std::vector<int> candidateIndices;  // Indices of points that passed filter
    std::vector<RayResult> filterHits;  // Hit results from filter phase
    std::vector<RayResult> refineHits;  // Hit results from refine phase
    std::vector<int> finalResults;      // Final mapping: point index -> polygon index (-1 if outside)
    size_t totalPoints;
    size_t filteredOut;
    size_t candidateCount;
    size_t insideCount;
    size_t outsideCount;
};

class FilterRefinePipeline {
public:
    FilterRefinePipeline(OptixContext& context, OptixPipelineManager& pipeline);
    
    // Execute filter phase: test points against filter geometry (typically bounding box)
    FilterRefineResult executeFilter(const GeometryData& filterGeometry,
                                     const PointData& pointData);
    
    // Execute refine phase: exact raytracing for candidate points
    FilterRefineResult executeRefine(const GeometryData& refineGeometry,
                                     const std::vector<float3>& candidatePoints,
                                     const std::vector<int>& candidateIndices,
                                     size_t totalPoints);
    
    // Execute both phases: filter then refine
    FilterRefineResult execute(const GeometryData& queryGeometry,
                               const PointData& pointData);
    
private:
    OptixContext& context_;
    OptixPipelineManager& pipeline_;
};

