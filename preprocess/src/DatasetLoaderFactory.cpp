#include "DatasetLoaderFactory.h"

#include <filesystem>

#include "PolygonDatasetLoader.h"
#include "ObjMeshDatasetLoader.h"
#include "DtMeshDatasetLoader.h"

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::create(DatasetType type) {
    switch (type) {
        case DatasetType::Polygon:
            return std::make_unique<PolygonDatasetLoader>();
        case DatasetType::Mesh:
            return std::make_unique<ObjMeshDatasetLoader>();
        case DatasetType::DtMesh:
            return std::make_unique<DtMeshDatasetLoader>();
        default:
            return nullptr;
    }
}

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::createFromPath(const std::string& path) {
    std::error_code ec;
    std::cout << "DEBUG: createFromPath called with: " << path << std::endl;
    std::cout << "DEBUG: is_regular_file: " << std::filesystem::is_regular_file(path, ec) << std::endl;
    std::cout << "DEBUG: extension: [" << std::filesystem::path(path).extension() << "]" << std::endl;
    
    if (std::filesystem::is_regular_file(path, ec)) {
        if (std::filesystem::path(path).extension() == ".obj") {
            std::cout << "DEBUG: Creating ObjMeshDatasetLoader" << std::endl;
            return std::make_unique<ObjMeshDatasetLoader>();
        } else if (std::filesystem::path(path).extension() == ".dt") {
            std::cout << "DEBUG: Creating DtMeshDatasetLoader" << std::endl;
            return std::make_unique<DtMeshDatasetLoader>();
        }
    }
    std::cout << "DEBUG: Falling back to PolygonDatasetLoader" << std::endl;
    return std::make_unique<PolygonDatasetLoader>();
}
