#include "DatasetLoaderFactory.h"

#include <filesystem>

#include "PolygonDatasetLoader.h"
#include "MeshDatasetLoader.h"

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::create(DatasetType type) {
    switch (type) {
        case DatasetType::Polygon:
            return std::make_unique<PolygonDatasetLoader>();
        case DatasetType::Mesh:
            return std::make_unique<MeshDatasetLoader>();
        default:
            return nullptr;
    }
}

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::createFromPath(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(path, ec) && std::filesystem::path(path).extension() == ".obj") {
        return std::make_unique<MeshDatasetLoader>();
    }
    return std::make_unique<PolygonDatasetLoader>();
}
