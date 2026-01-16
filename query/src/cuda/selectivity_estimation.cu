#include "selectivity_estimation.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__device__ int get_count(const int* data, int nx, int ny, int nz, int ix, int iy, int iz) {
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return 0;
    return data[iz * (nx * ny) + iy * nx + ix];
}

__device__ int3 get_indices(float3 p, float3 minB, float3 cellS) {
    return {
        (int)((p.x - minB.x) / cellS.x),
        (int)((p.y - minB.y) / cellS.y),
        (int)((p.z - minB.z) / cellS.z)
    };
}

// Estimates intersection pairs by summing (FacesA * FacesB) for overlapping cells
__global__ void estimate_kernel(
    const int* dA_f, int3 dimsA, float3 minBA, float3 cellSA,
    const int* dB_f, int3 dimsB, float3 minBB, float3 cellSB,
    unsigned long long* d_result
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCellsA = dimsA.x * dimsA.y * dimsA.z;
    if (idx >= totalCellsA) return;

    // Decode index for A
    int temp = idx;
    int ix = temp % dimsA.x; temp /= dimsA.x;
    int iy = temp % dimsA.y; 
    int iz = temp / dimsA.y;

    int countA = dA_f[idx];
    if (countA == 0) return;

    // Center of cell A in world space
    float3 centerA = {
        minBA.x + (ix + 0.5f) * cellSA.x,
        minBA.y + (iy + 0.5f) * cellSA.y,
        minBA.z + (iz + 0.5f) * cellSA.z
    };

    // Find corresponding cell in B
    int3 idxB = get_indices(centerA, minBB, cellSB);
    
    // Check simple point containment
    int countB = get_count(dB_f, dimsB.x, dimsB.y, dimsB.z, idxB.x, idxB.y, idxB.z);
    
    if (countB > 0) {
        // Multiply face counts to estimate potential intersection pairs
        atomicAdd(d_result, (unsigned long long)countA * countB);
    }
}

size_t estimateSelectivityGPU(const EulerHistogram& histA, const EulerHistogram& histB) {
    if (histA.f_counts.empty() || histB.f_counts.empty()) {
        std::cerr << "Warning: Empty Euler Histogram(s) provided for estimation." << std::endl;
        return 0;
    }

    int* dA = nullptr; 
    int* dB = nullptr;
    unsigned long long* dRes = nullptr;

    // Use object counts ONLY if available for BOTH. Consistent semantics are required.
    bool usingObjectCounts = !histA.object_counts.empty() && !histB.object_counts.empty();
    
    const std::vector<int>& dataA = usingObjectCounts ? histA.object_counts : histA.f_counts;
    const std::vector<int>& dataB = usingObjectCounts ? histB.object_counts : histB.f_counts;

    auto printStats = [](const std::string& name, const std::vector<int>& data) {
        size_t filled = 0;
        int maxVal = 0;
        unsigned long long sum = 0;
        for (int v : data) {
            if (v > 0) {
                filled++;
                sum += v;
                if (v > maxVal) maxVal = v;
            }
        }
        double occupancy = (double)filled / data.size() * 100.0;
        double avg = filled > 0 ? (double)sum / filled : 0.0;
        std::cout << "  " << name << " Stats: " << filled << "/" << data.size() 
                  << " cells filled (" << occupancy << "%), Avg/Filled: " << avg 
                  << ", Max: " << maxVal << std::endl;
    };

    if (usingObjectCounts) {
        std::cout << "  [Estimation Mode] Using DIRECT OBJECT COUNTS (Precomputed)" << std::endl;
    } else {
        std::cout << "  [Estimation Mode] Using TRIANGLE COUNTS (Fallback)" << std::endl;
    }
    
    printStats("Grid A", dataA);
    printStats("Grid B", dataB);

    size_t szA = dataA.size() * sizeof(int);
    size_t szB = dataB.size() * sizeof(int);
    
    cudaError_t err;
    err = cudaMalloc(&dA, szA);
    if(err != cudaSuccess) { std::cerr << "CUDA Malloc failed for histA" << std::endl; return 0; }
    
    err = cudaMalloc(&dB, szB);
    if(err != cudaSuccess) { cudaFree(dA); std::cerr << "CUDA Malloc failed for histB" << std::endl; return 0; }
    
    err = cudaMalloc(&dRes, sizeof(unsigned long long));
    
    cudaMemcpy(dA, dataA.data(), szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, dataB.data(), szB, cudaMemcpyHostToDevice);
    cudaMemset(dRes, 0, sizeof(unsigned long long));

    int blockSize = 256;
    int numBlocks = (static_cast<int>(histA.f_counts.size()) + blockSize - 1) / blockSize;

    int3 dimsA = {histA.nx, histA.ny, histA.nz};
    int3 dimsB = {histB.nx, histB.ny, histB.nz};

    estimate_kernel<<<numBlocks, blockSize>>>(
        dA, dimsA, histA.minBound, histA.cellSize,
        dB, dimsB, histB.minBound, histB.cellSize,
        dRes
    );
    
    unsigned long long hRes = 0;
    cudaMemcpy(&hRes, dRes, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dRes);
    
    return (size_t)hRes;
}
