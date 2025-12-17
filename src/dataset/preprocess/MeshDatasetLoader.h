#pragma once

#include <string>
#include <memory>

#include "IDatasetLoader.h"
#include "../common/Geometry.h"

class MeshDatasetLoader : public IDatasetLoader {
public:
    GeometryData load(const std::string& filePath) override;
};
