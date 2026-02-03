# Mesh Overlap Query Debug Analysis

## Overview
This document summarizes the investigation into performance degradation observed in the `mesh_overlap` query, specifically when using multi-hit support per edge. The investigation revealed that the performance drop is caused by numerical precision issues leading to infinite ray tracing loops.

## Key Findings

### 1. The "Stuck Ray" Phenomenon
The primary cause of the performance deterioration is rays getting "stuck" hitting the same triangle repeatedly.
*   **97.22%** of all recorded partial intersections were redundant hits on the exact same triangle.
*   Approximately **35%** of all ray traces hit the hard loop limit (`kMaxIterations = 100`) because they failed to advance through the scene.

### 2. Root Cause: Floating Point Precision
The issue stems from the logic used to advance the ray origin after a hit:
```cpp
origin = origin + direction * (t + epsilon);
```
*   **Scene Scale**: The geometry coordinates are relatively large (approx. `3000.0` on the Z-axis).
*   **Machine Epsilon**: At a magnitude of `3000.0`, the spacing between representable single-precision floats (machine epsilon) is approximately `0.00024`.
*   **Step Size**: The ray intersection distance `t` plus the fixed `epsilon` (`1e-6`) resulted in a step size of approximately `0.000015`.
*   **Failure**: The step size is **smaller than the floating point resolution** at this coordinate scale.
    *   `3000.0 + 0.000015 == 3000.0` (in float32)
    *   The ray origin effectively never changes.
    *   OptiX re-traces from the exact same point, finds the same intersection, and the loop repeats indefinitely.

### 3. Statistical Evidence
Data collected from a debug run with `check_n_geometry` and `check_v_geometry`:
*   **Total Hits recorded**: 637,179
*   **Redundant "Stuck" Hits**: 619,443 (97.22%)
*   **Traces hitting 50-100 times**: 6,196

Example of a stuck trace:
```
[WARN] Immediate Self-Intersection! Trace(Pass=1, Tri=894, Edge=1) Iter=1 t=7.82127e-05 Tri=115347
[WARN] Immediate Self-Intersection! Trace(Pass=1, Tri=894, Edge=1) Iter=2 t=7.82127e-05 Tri=115347
...
```
The `t` value remains identical because the origin never updates.

## Recommendations

To resolve this, the ray advancement logic must be robust to floating point/scene scale issues.

### 1. Relative Epsilon (Dynamic)
Scale the epsilon based on the magnitude of the origin coordinates.
```cpp
float max_comp = fmaxf(fabs(origin.x), fmaxf(fabs(origin.y), fabs(origin.z)));
float robust_epsilon = 1e-6f * fmaxf(1.0f, max_comp);
```

### 2. Ray Offsetting (Safe Step)
Ensure the new origin is mathematically distinct from the previous one, potentially using `nextafter` or a dedicated robust ray offsetting function (like those used in PBRT or other ray tracers to avoid self-intersection acne/looping).

### 3. Geometric Epsilon
Given the large coordinates, a larger fixed epsilon (e.g., `1e-3` or `1e-4`) might be necessary if a dynamic solution is not preferred, though this risks missing valid close-range intersections.
