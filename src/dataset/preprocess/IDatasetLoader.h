#pragma once

#include <string>
#include <memory>

#include "../common/Geometry.h" // For GeometryData

// Interface for interchangeable dataset loaders
class IDatasetLoader {
public:
    virtual ~IDatasetLoader() = default;
    // Load geometry from a dataset path (file for polygons, directory for meshes)
    virtual GeometryData load(const std::string& path) = 0;
};
