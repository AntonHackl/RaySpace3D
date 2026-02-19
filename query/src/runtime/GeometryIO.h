#pragma once

#include <string>
#include "Geometry.h"

// Load preprocessed geometry from a binary file created by preprocess_dataset
GeometryData loadGeometryFromFile(const std::string& geometryFilePath);
