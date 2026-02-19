#include "FilterRefine.h"
#include "../optix/OptixHelpers.h"
#include "../optix/OptixAccelerationStructure.h"
#include <iostream>

FilterRefinePipeline::FilterRefinePipeline(OptixContext& context, OptixPipelineManager& pipeline)
    : context_(context), pipeline_(pipeline) {
}

FilterRefineResult FilterRefinePipeline::executeFilter(const GeometryData& filterGeometry,
                                                       const PointData& pointData) {
    FilterRefineResult result;
    result.totalPoints = pointData.numPoints;
    
    GeometryUploader filterUploader;
    filterUploader.upload(filterGeometry);
    
    OptixAccelerationStructure filterAS(context_, filterUploader);
    filterAS.build();
    
    RayLauncher launcher(pipeline_, filterUploader);
    launcher.uploadRays(pointData.positions.data(), pointData.numPoints);
    
    launcher.launch(filterAS, static_cast<int>(pointData.numPoints));
    
    ResultProcessor processor;
    result.filterHits = processor.compactAndDownload(launcher.getResultBuffer(), 
                                                      static_cast<int>(pointData.numPoints));
    
    result.candidateIndices.reserve(result.filterHits.size());
    for (const auto& hit : result.filterHits) {
        result.candidateIndices.push_back(hit.ray_id);
    }
    
    result.candidateCount = result.candidateIndices.size();
    result.filteredOut = result.totalPoints - result.candidateCount;
    
    return result;
}

FilterRefineResult FilterRefinePipeline::executeRefine(const GeometryData& refineGeometry,
                                                        const std::vector<float3>& candidatePoints,
                                                        const std::vector<int>& candidateIndices,
                                                        size_t totalPoints) {
    FilterRefineResult result;
    result.totalPoints = totalPoints;
    result.candidateIndices = candidateIndices;
    result.candidateCount = candidatePoints.size();
    
    if (candidatePoints.empty()) {
        result.filteredOut = totalPoints;
        result.outsideCount = totalPoints;
        result.insideCount = 0;
        result.finalResults.assign(totalPoints, -1);
        return result;
    }
    
    GeometryUploader refineUploader;
    refineUploader.upload(refineGeometry);
    
    OptixAccelerationStructure refineAS(context_, refineUploader);
    refineAS.build();
    
    RayLauncher launcher(pipeline_, refineUploader);
    launcher.uploadRays(candidatePoints);
    
    launcher.launch(refineAS, static_cast<int>(candidatePoints.size()));
    
    ResultProcessor processor;
    result.refineHits = processor.compactAndDownload(launcher.getResultBuffer(),
                                                      static_cast<int>(candidatePoints.size()));
    
    result.finalResults = processor.mapToOriginalIndices(result.refineHits, candidateIndices, totalPoints);
    
    result.insideCount = result.refineHits.size();
    result.outsideCount = totalPoints - result.insideCount;
    
    return result;
}

FilterRefineResult FilterRefinePipeline::execute(const GeometryData& queryGeometry,
                                                  const PointData& pointData) {
    // Phase 1: Filter - compute bounding box and filter points
    BoundingBox bbox = BoundingBox::computeFromGeometry(queryGeometry);
    BoundingBox::BoxMesh boxMesh = bbox.createBoxMesh();
    
    GeometryData filterGeometry;
    filterGeometry.vertices.assign(boxMesh.vertices.begin(), boxMesh.vertices.end());
    filterGeometry.indices.assign(boxMesh.indices.begin(), boxMesh.indices.end());
    filterGeometry.triangleToObject.assign(boxMesh.triangleToObject.begin(), boxMesh.triangleToObject.end());
    filterGeometry.totalTriangles = boxMesh.indices.size();
    
    FilterRefineResult filterResult = executeFilter(filterGeometry, pointData);
    
    if (filterResult.candidateCount == 0) {
        // No candidates, all points filtered out
        filterResult.finalResults.assign(pointData.numPoints, -1);
        filterResult.outsideCount = pointData.numPoints;
        filterResult.insideCount = 0;
        return filterResult;
    }
    
    // Phase 2: Refine - prepare candidate points
    std::vector<float3> candidatePoints;
    candidatePoints.reserve(filterResult.candidateCount);
    for (int idx : filterResult.candidateIndices) {
        if (idx >= 0 && static_cast<size_t>(idx) < pointData.numPoints) {
            candidatePoints.push_back(pointData.positions[idx]);
        }
    }
    
    FilterRefineResult refineResult = executeRefine(queryGeometry, candidatePoints,
                                                     filterResult.candidateIndices, pointData.numPoints);
    
    FilterRefineResult result = filterResult;
    result.refineHits = refineResult.refineHits;
    result.finalResults = refineResult.finalResults;
    result.insideCount = refineResult.insideCount;
    result.outsideCount = refineResult.outsideCount;
    
    return result;
}

