#pragma once
#include "../../common/include/Geometry.h"

// Returns estimated number of intersecting primitives (or pairs)
// Returns 0 if histograms are invalid.
size_t estimateSelectivityGPU(const EulerHistogram& histA, const EulerHistogram& histB);
