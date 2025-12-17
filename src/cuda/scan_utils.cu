#include "scan_utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

extern "C" {

int exclusive_scan_gpu(const int* d_input, int* d_output, int num_elements) {
    if (num_elements == 0) {
        return 0;
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Wrap device pointers
    thrust::device_ptr<const int> input_begin(d_input);
    thrust::device_ptr<const int> input_end = input_begin + num_elements;
    thrust::device_ptr<int> output_begin(d_output);
    
    // Perform exclusive scan (prefix sum)
    thrust::exclusive_scan(thrust::device, input_begin, input_end, output_begin);
    cudaDeviceSynchronize();
    
    // Calculate total sum using reduce (much faster than two memcpy calls)
    int total = thrust::reduce(thrust::device, input_begin, input_end, 0);
    cudaDeviceSynchronize();
    
    return total;
}

} // extern "C"
