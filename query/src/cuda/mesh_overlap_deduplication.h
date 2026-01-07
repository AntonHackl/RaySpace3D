#pragma once

#include "common.h"
#include <cuda_runtime.h>

extern "C" {
int merge_and_deduplicate_gpu(
    const MeshOverlapResult* d_results1, int num_results1,
    const MeshOverlapResult* d_results2, int num_results2,
    MeshOverlapResult* d_merged_output  // Should be allocated with size >= (num_results1 + num_results2)
);

} // extern "C"
