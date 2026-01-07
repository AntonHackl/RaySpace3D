#pragma once

#include <memory>
#include <string>

#include "IDatasetLoader.h"

enum class DatasetType {
    Polygon,
    Mesh,
    DtMesh
};

class DatasetLoaderFactory {
public:
    static std::unique_ptr<IDatasetLoader> create(DatasetType type);
    // Heuristic: decide based on path (.obj file -> Mesh, otherwise -> Polygon)
    static std::unique_ptr<IDatasetLoader> createFromPath(const std::string& path);
};
