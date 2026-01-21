# Selectivity Estimation in RaySpace3D

This document outlines the methodology used for estimating intersection selectivity (result size) in the RaySpace3D intersection application. The logic is implemented primarily in `query/src/cuda/estimated_intersection.cu` and `query/src/applications/rayspace_intersection_estimated_main.cpp`.

## Overview

The estimation process predicts the number of intersecting pairs between two 3D datasets without performing exact geometry tests. It relies on a probabilistic model applied to pre-computed grid statistics.

## Grid-Based Approach

Both datasets are pre-processed into a uniform grid structure. For each cell in the grid, the following statistics are stored (defined in `common/include/GridCell.h`):

*   **`TouchCount`**: The number of objects interacting with (touching) the cell.
*   **`AvgSizeMean`**: The average size of objects in the cell, calculated as the mean of their dimensions $((Width + Height + Depth) / 3)$.
*   **`VolRatio`**: The average "fullness" of the objects, calculated as the ratio of the object's mesh volume to its bounding box volume.

## Probabilistic Model

The estimation kernel (`estimateKernel`) iterates over every cell index $i$ in the grid. If both datasets have objects touching the cell ($TouchCount_A > 0$ and $TouchCount_B > 0$), an intersection probability is calculated.

### 1. Minkowski Volume Approximation
The probability of intersection is based on the [Minkowski Sum](https://en.wikipedia.org/wiki/Minkowski_addition) concept. We approximate the combined size of objects from both datasets:

$$ 
S_{combined} = AvgSizeMean_A + AvgSizeMean_B + \epsilon
$$

where $\epsilon$ is a small constant (default 0.001) to prevent underestimation for very small objects.

The "Minkowski Volume" is then approximated as a cube:

$$
V_{minkowski} = (S_{combined})^3
$$

The base probability $P_{base}$ is the ratio of this volume to the cell's volume:

$$
P_{base} = \frac{V_{minkowski}}{V_{cell}}
$$

### 2. Shape Correction ($ \gamma $)
To account for object shapes (sparse vs. dense), a correction factor is applied derived from the Volume Ratios.

We compute the geometric mean of the volume ratios from both datasets:

$$
R_{combined} = \sqrt{VolRatio_A \times VolRatio_B}
$$

The shape correction factor is controlled by a user-configurable parameter $\gamma$ (Gamma):

$$
F_{shape} = (R_{combined})^\gamma
$$

*   $\gamma$ allows tuning the sensitivity to object density.
*   Default $\gamma$ is 0.8.

### 3. Final Probability
The final probability $P$ for the cell is:

$$
P = \min \left( P_{base} \times F_{shape}, \quad 1.0 \right)
$$

## Estimation Calculation

The estimated number of intersections for a specific cell $i$ is calculated as:

$$
E_i = TouchCount_A \times TouchCount_B \times P
$$

The total estimated result size is the sum over all cells:

$$
E_{total} = \sum_{i=0}^{N} E_i
$$

## Execution Flow

1.  **Data Loading**: The application loads the geometry and grid data for both meshes.
2.  **Kernel Launch**: A CUDA kernel (`estimateKernel`) is launched with one thread per grid cell.
3.  **Local Compute**: Each thread computes $E_i$ for its assigned cell.
4.  **Reduction**: A `thrust::reduce` operation sums all $E_i$ values to produce the final result.

## Parameters

*   **`--gamma <float>`**: Controls the impact of the volume ratio shape correction. Higher values reduce the probability for sparse objects. Default: `0.8`.
*   **`--epsilon <float>`**: A small bias added to the size calculation. Default: `0.001`.
