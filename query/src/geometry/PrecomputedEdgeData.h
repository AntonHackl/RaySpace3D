#pragma once

#include "Geometry.h"

struct EdgeMeshData {
    float3* d_edge_starts;
    float3* d_edge_ends;
    int* d_source_objects;
    int* d_source_object_offsets;
    int* d_source_object_counts;
    int num_edges;

    EdgeMeshData()
        : d_edge_starts(nullptr),
          d_edge_ends(nullptr),
          d_source_objects(nullptr),
          d_source_object_offsets(nullptr),
          d_source_object_counts(nullptr),
          num_edges(0) {}
};

class PrecomputedEdgeData {
public:
    static EdgeMeshData uploadFromGeometry(const GeometryData& geometry);
    static void freeEdgeData(EdgeMeshData& edgeData);
};
