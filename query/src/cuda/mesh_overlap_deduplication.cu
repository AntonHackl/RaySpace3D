#include "mesh_overlap_deduplication.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

// Convert pair to single 64-bit key for faster sorting
__device__ __host__ inline unsigned long long pair_to_key(int id1, int id2) {
    return (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);
}

__device__ __host__ inline void key_to_pair(unsigned long long key, int& id1, int& id2) {
    id1 = static_cast<int>(key >> 32);
    id2 = static_cast<int>(key & 0xFFFFFFFF);
}

// Kernel to convert pairs to keys
__global__ void pairs_to_keys_kernel(const MeshOverlapResult* pairs, unsigned long long* keys, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        keys[idx] = pair_to_key(pairs[idx].object_id_mesh1, pairs[idx].object_id_mesh2);
    }
}

// Kernel to convert unique keys back to pairs
__global__ void keys_to_pairs_kernel(const unsigned long long* keys, MeshOverlapResult* pairs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        key_to_pair(keys[idx], pairs[idx].object_id_mesh1, pairs[idx].object_id_mesh2);
    }

}

__global__ void compact_hash_pairs_kernel(
    const unsigned long long* hash_table, 
    int capacity, 
    MeshOverlapResult* output, 
    int* count_out, 
    int max_output_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < capacity) {
        unsigned long long key = hash_table[idx];
        if (key != 0xFFFFFFFFFFFFFFFFULL) {
            int pos = atomicAdd(count_out, 1);
            if (pos < max_output_size) {
                 int id1 = static_cast<int>(key >> 32);
                 int id2 = static_cast<int>(key & 0xFFFFFFFF);
                 output[pos].object_id_mesh1 = id1;
                 output[pos].object_id_mesh2 = id2;
            }
        }
    }
}

extern "C" {

int merge_and_deduplicate_gpu(
    const MeshOverlapResult* d_results1, int num_results1,
    const MeshOverlapResult* d_results2, int num_results2,
    MeshOverlapResult* d_merged_output
) {
    int total_results = num_results1 + num_results2;
    
    if (total_results == 0) {
        return 0;
    }
    
    unsigned long long* d_keys;
    cudaMalloc(&d_keys, total_results * sizeof(unsigned long long));
    
    int threads = 256;
    int blocks = (total_results + threads - 1) / threads;
    
    // Step 1: Copy and convert both result arrays to keys
    if (num_results1 > 0) {
        cudaMemcpy(d_merged_output, d_results1, 
                   num_results1 * sizeof(MeshOverlapResult), 
                   cudaMemcpyDeviceToDevice);
    }
    if (num_results2 > 0) {
        cudaMemcpy(d_merged_output + num_results1, d_results2, 
                   num_results2 * sizeof(MeshOverlapResult), 
                   cudaMemcpyDeviceToDevice);
    }
    
    pairs_to_keys_kernel<<<blocks, threads>>>(d_merged_output, d_keys, total_results);
    cudaDeviceSynchronize();
    
    // Step 2: Sort keys (much faster than sorting structs)
    thrust::device_ptr<unsigned long long> key_begin(d_keys);
    thrust::device_ptr<unsigned long long> key_end = key_begin + total_results;
    thrust::sort(thrust::device, key_begin, key_end);
    
    // Step 3: Remove duplicate keys
    thrust::device_ptr<unsigned long long> new_key_end = 
        thrust::unique(thrust::device, key_begin, key_end);
    
    int num_unique = new_key_end - key_begin;
    
    // Step 4: Convert unique keys back to pairs
    int unique_blocks = (num_unique + threads - 1) / threads;
    keys_to_pairs_kernel<<<unique_blocks, threads>>>(d_keys, d_merged_output, num_unique);
    cudaDeviceSynchronize();
    
    cudaFree(d_keys);
    
    return num_unique;
}

int compact_hash_table(
    const unsigned long long* d_hash_table, int table_size,
    MeshOverlapResult* d_output, int max_output_size
) {
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    int threads = 256;
    int blocks = (table_size + threads - 1) / threads;
    
    compact_hash_pairs_kernel<<<blocks, threads>>>(
        d_hash_table, table_size, d_output, d_count, max_output_size
    );
    cudaDeviceSynchronize();
    
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    
    return (h_count < max_output_size) ? h_count : max_output_size;
}

} // extern "C"
