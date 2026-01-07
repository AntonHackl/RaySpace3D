#pragma once

#include <string>
#include "Geometry.h"

// Load preprocessed geometry from a text file created by preprocess_dataset
GeometryData loadGeometryFromFile(const std::string& geometryFilePath);
