# Selectivity Estimation in RaySpace3D

This document outlines the methodology used for estimating intersection selectivity (result size) in the RaySpace3D intersection application. The logic is implemented primarily in `query/src/cuda/estimated_intersection.cu` and `query/src/applications/rayspace_intersection_estimated_main.cpp`.

## Overview

The estimation process predicts the number of intersecting pairs between two 3D datasets without performing exact geometry tests. It relies on a probabilistic model applied to pre-computed grid statistics.

## Preprocessing and Grid Statistics

Before the query, each dataset is discretized into a uniform 3D grid during the preprocessing phase (see `preprocess/src/preprocess_dataset.cpp`).

### 1. Object Feature Extraction
For each unique object in the mesh, the following physical properties are calculated:
*   **Dimensions**: The width, height, and depth from its Axis-Aligned Bounding Box (AABB).
*   **Mesh Volume ($V_{mesh}$)**: Calculated using the divergence theorem (summed signed volumes of tetrahedrons formed by the mesh triangles).
*   **AABB Volume ($V_{aabb}$)**: $Width \times Height \times Depth$.
*   **Volume Ratio ($R_{vol}$)**: $V_{mesh} / V_{aabb}$, representing the "fullness" or density of the object.
*   **Average Size ($S_{avg}$)**: $(Width + Height + Depth) / 3$.

### 2. Grid Mapping
Objects are mapped to the grid using their AABB:
*   **`CenterCount`**: Incremented for the single cell containing the object's AABB center.
*   **`TouchCount`**: Incremented for every cell that overlaps with the object's AABB.
*   **`AvgSizeMean` & `VolRatio`**: For each cell an object touches, its $S_{avg}$ and $R_{vol}$ are added to running sums for that cell.

### 3. Final Grid Cell Statistics
After all objects are processed, the accumulated values in each cell are normalized by the cell's `TouchCount` to produce the final statistics stored in `common/include/GridCell.h`:

*   **`TouchCount`**: Total number of objects interacting with the cell.
*   **`AvgSizeMean`**: The average $S_{avg}$ of all objects touching the cell.
*   **`VolRatio`**: The average $R_{vol}$ of all objects touching the cell.

## Probabilistic Model

The estimation kernel (`estimateKernel` in `query/src/cuda/estimated_intersection.cu`) iterates over every cell index $i$ in the grid. If both datasets have objects touching the cell ($TouchCount_A > 0$ and $TouchCount_B > 0$), an intersection potential is calculated.

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

## Estimation Calculation and Normalization

The total estimation is a two-step process: calculating raw potential pairs and then normalizing for object replication across cells.

### 1. Raw Intersection Potential
For each cell $i$, the kernel calculates:
$$ E_i = TouchCount_A \times TouchCount_B \times P $$
The sum of these potentials is the raw estimate:
$$ E_{raw} = \sum_{i=0}^{N} E_i $$

### 2. Normalization (Replication Correction)
Because objects are counted in every cell they touch, a large object pair might be detected as intersecting in multiple cells. To correct for this overcounting, we divide by a replication factor $\alpha$.

First, we calculate the global average object size for each dataset ($\bar{S}_A, \bar{S}_B$) by averaging the `AvgSizeMean` of all occupied cells. Then we compute:
$$ \alpha = \max \left( \frac{(\bar{S}_A + \bar{S}_B)^3}{V_{cell}}, \quad 1.0 \right) $$

The final predicted number of unique intersecting pairs is:
$$ E_{final} = \frac{E_{raw}}{\alpha} $$

## Application in Query Configuration

The final estimate $E_{final}$ is used to optimize the subsequent exact intersection query in `rayspace_intersection_estimated_main.cpp`:

*   **Hash Table Sizing**: To minimize collisions while saving memory, the deduplication hash table is sized based on the estimate:
    $$ \text{TargetSize} = \frac{E_{final}}{0.5} \quad (\text{Target Load Factor of 0.5}) $$
    The actual size used is the next power of two of this target, clamped between $1,024$ and $1,073,741,824$.

## Execution Flow

1.  **Data Loading**: Loads geometry and pre-computed grid binary data.
2.  **Kernel Launch**: `estimateKernel` computes raw per-cell potentials $E_i$ on the GPU.
3.  **Reduction**: `thrust::reduce` sums per-cell values to produce $E_{raw}$.
4.  **Global Analysis**: The CPU calculates global average sizes and the replication factor $\alpha$.
5.  **Final Estimation**: $E_{final}$ is produced and used to allocate the deduplication hash table.
6.  **Intersection Query**: (Optional) Performs the actual OptiX-based intersection tests using the optimized hash table.

## Parameters

*   **`--gamma <float>`**: Controls the impact of the volume ratio shape correction. Higher values reduce the probability for sparse objects. Default: `0.8`.
*   **`--epsilon <float>`**: A small bias added to the size calculation. Default: `0.001`.
