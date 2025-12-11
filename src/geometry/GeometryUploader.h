#pragma once

#include "../common.h"
#include "../dataset/common/Geometry.h"
#include "../optix/OptixHelpers.h"
#include <vector>

class GeometryUploader {
public:
    GeometryUploader();
    ~GeometryUploader();
    
    // Disable copy construction and assignment
    GeometryUploader(const GeometryUploader&) = delete;
    GeometryUploader& operator=(const GeometryUploader&) = delete;
    
    // Upload geometry data to GPU
    void upload(const GeometryData& geometry);
    
    // Upload geometry from vectors directly
    void upload(const std::vector<float3>& vertices,
                const std::vector<uint3>& indices,
                const std::vector<int>& triangleToObject);
    
    // Get device pointers
    float3* getVertices() const { return d_vertices_; }
    uint3* getIndices() const { return d_indices_; }
    int* getTriangleToObject() const { return d_triangle_to_object_; }
    
    size_t getNumVertices() const { return num_vertices_; }
    size_t getNumIndices() const { return num_indices_; }
    size_t getNumTrianglesToObject() const { return num_triangle_to_object_; }
    
    // Free GPU memory
    void free();
    
    bool isUploaded() const { return d_vertices_ != nullptr; }
    
private:
    float3* d_vertices_;
    uint3* d_indices_;
    int* d_triangle_to_object_;
    size_t num_vertices_;
    size_t num_indices_;
    size_t num_triangle_to_object_;
    
    void freeInternal();
};

