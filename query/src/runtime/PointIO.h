#pragma once

#include <string>
#include "Geometry.h"

// Load point dataset (WKT POINT lines or simple xyz/xy list)
PointData loadPointDataset(const std::string& pointDatasetPath);
