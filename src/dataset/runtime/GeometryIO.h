#pragma once

#include <string>
#include "../common/Geometry.h"

// Load preprocessed geometry from a text file created by preprocess_dataset
GeometryData loadGeometryFromFile(const std::string& geometryFilePath);
