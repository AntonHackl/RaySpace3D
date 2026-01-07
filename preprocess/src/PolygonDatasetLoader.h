#pragma once

#include <string>
#include <memory>

#include "IDatasetLoader.h"
#include "Geometry.h"

class PolygonDatasetLoader : public IDatasetLoader {
public:
    GeometryData load(const std::string& wktFilePath) override;
};
