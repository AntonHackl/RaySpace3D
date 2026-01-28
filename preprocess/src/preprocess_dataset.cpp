// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
// #include "Geometry.h"
#include "../../common/include/Geometry.h"
#include "../../common/include/BinaryIO.h"
#include "DatasetLoaderFactory.h"
#include "timer.h"

void writeGeometryDataToFile(const GeometryData& geometry, const std::string& filename) {
    if (RaySpace::IO::writeBinaryFile(filename, geometry)) {
        std::cout << "Geometry data written to: " << filename << " (Binary)" << std::endl;
    }
}

#include <cmath>
#include <algorithm>

struct ObjectStats {
    float3 minB = {1e30f, 1e30f, 1e30f};
    float3 maxB = {-1e30f, -1e30f, -1e30f};
    double volume = 0.0;
};

void generateGridStats(GeometryData& geometry, GridData& grid, int resolution, float worldSize = -1.0f) {
    grid.resolution = { (unsigned int)resolution, (unsigned int)resolution, (unsigned int)resolution };
    
    // Compute bounds from geometry if not specified or as a safety
    float3 minB = {1e30f, 1e30f, 1e30f};
    float3 maxB = {-1e30f, -1e30f, -1e30f};
    for (const auto& v : geometry.vertices) {
        minB.x = std::min(minB.x, v.x);
        minB.y = std::min(minB.y, v.y);
        minB.z = std::min(minB.z, v.z);
        maxB.x = std::max(maxB.x, v.x);
        maxB.y = std::max(maxB.y, v.y);
        maxB.z = std::max(maxB.z, v.z);
    }
    
    // If worldSize is positive, we use [0, worldSize] as requested legacy behavior
    // but better to use actual bounds
    if (worldSize <= 0) {
        grid.minBound = minB;
        grid.maxBound = maxB;
    } else {
        grid.minBound = {0.0f, 0.0f, 0.0f};
        grid.maxBound = {worldSize, worldSize, worldSize};
    }
    
    float3 range = {grid.maxBound.x - grid.minBound.x, grid.maxBound.y - grid.minBound.y, grid.maxBound.z - grid.minBound.z};
    float3 cellSize = {range.x / resolution, range.y / resolution, range.z / resolution};
    
    grid.cells.assign(resolution * resolution * resolution, {0, 0, 0.0f, 0.0f});
    grid.hasGrid = true;
    
    std::unordered_map<int, ObjectStats> objects;
    
    // Pass 1: AABB
    for (size_t i = 0; i < geometry.indices.size(); ++i) {
        int objId = geometry.triangleToObject[i];
        const auto& idx = geometry.indices[i];
        // Indices are likely 0-based index into vertices
        if (idx.x >= geometry.vertices.size() || idx.y >= geometry.vertices.size() || idx.z >= geometry.vertices.size()) continue;

        const float3& v0 = geometry.vertices[idx.x];
        const float3& v1 = geometry.vertices[idx.y];
        const float3& v2 = geometry.vertices[idx.z];
        
        ObjectStats& stats = objects[objId];
        stats.minB.x = std::min({stats.minB.x, v0.x, v1.x, v2.x});
        stats.minB.y = std::min({stats.minB.y, v0.y, v1.y, v2.y});
        stats.minB.z = std::min({stats.minB.z, v0.z, v1.z, v2.z});
        stats.maxB.x = std::max({stats.maxB.x, v0.x, v1.x, v2.x});
        stats.maxB.y = std::max({stats.maxB.y, v0.y, v1.y, v2.y});
        stats.maxB.z = std::max({stats.maxB.z, v0.z, v1.z, v2.z});
    }

    // Pass 2: Volume 
    // We compute volume relative to AABB center to improve precision
    for (size_t i = 0; i < geometry.indices.size(); ++i) {
        int objId = geometry.triangleToObject[i];
        const auto& idx = geometry.indices[i];
        if (idx.x >= geometry.vertices.size() || idx.y >= geometry.vertices.size() || idx.z >= geometry.vertices.size()) continue;
        
        const float3& v0 = geometry.vertices[idx.x];
        const float3& v1 = geometry.vertices[idx.y];
        const float3& v2 = geometry.vertices[idx.z];
        
        ObjectStats& stats = objects[objId];
        float3 center = {(stats.minB.x + stats.maxB.x)*0.5f, (stats.minB.y + stats.maxB.y)*0.5f, (stats.minB.z + stats.maxB.z)*0.5f};
        
        // Translate to center
        float3 t0 = {v0.x - center.x, v0.y - center.y, v0.z - center.z};
        float3 t1 = {v1.x - center.x, v1.y - center.y, v1.z - center.z};
        float3 t2 = {v2.x - center.x, v2.y - center.y, v2.z - center.z};
        
        // Signed Volume: dot(cross(t0, t1), t2)
        // cross(t0, t1)
        float cx = t0.y * t1.z - t0.z * t1.y;
        float cy = t0.z * t1.x - t0.x * t1.z;
        float cz = t0.x * t1.y - t0.y * t1.x;
        
        stats.volume += (cx * t2.x + cy * t2.y + cz * t2.z);
    }
    
    for (auto& kv : objects) {
        ObjectStats& stats = kv.second;
        float meshVol = std::abs((float)stats.volume) / 6.0f;
        
        float width = stats.maxB.x - stats.minB.x;
        float height = stats.maxB.y - stats.minB.y;
        float depth = stats.maxB.z - stats.minB.z;
        float aabbVol = width * height * depth;
        if (aabbVol < 1e-9f) aabbVol = 1e-9f; // Avoid div/0
        
        float ratio = meshVol / aabbVol;
        // Clamp ratio to [0, 1] as it can be > 1 if concave/errors or 1.000001
        if (ratio > 1.0f) ratio = 1.0f; 

        // Center
        float3 center = {(stats.minB.x + stats.maxB.x) * 0.5f,
                         (stats.minB.y + stats.maxB.y) * 0.5f,
                         (stats.minB.z + stats.maxB.z) * 0.5f};
        
        float avgSize = (width + height + depth) / 3.0f;
        
        // Anchor Cell
        int cx = (int)((center.x - grid.minBound.x) / cellSize.x);
        int cy = (int)((center.y - grid.minBound.y) / cellSize.y);
        int cz = (int)((center.z - grid.minBound.z) / cellSize.z);
        
        // Boundary check
        if (cx >= 0 && cx < resolution && cy >= 0 && cy < resolution && cz >= 0 && cz < resolution) {
            int idx = (cz * resolution + cy) * resolution + cx;
            grid.cells[idx].CenterCount++;
        }
        
        // Touch Cells (Conservative Rasterization/Overlap)
        int minCx = std::max(0, (int)((stats.minB.x - grid.minBound.x) / cellSize.x));
        int minCy = std::max(0, (int)((stats.minB.y - grid.minBound.y) / cellSize.y));
        int minCz = std::max(0, (int)((stats.minB.z - grid.minBound.z) / cellSize.z));
        
        int maxCx = std::min(resolution - 1, (int)((stats.maxB.x - grid.minBound.x - 1e-5f) / cellSize.x)); 
        int maxCy = std::min(resolution - 1, (int)((stats.maxB.y - grid.minBound.y - 1e-5f) / cellSize.y));
        int maxCz = std::min(resolution - 1, (int)((stats.maxB.z - grid.minBound.z - 1e-5f) / cellSize.z));
        
        for (int z = minCz; z <= maxCz; ++z) {
            for (int y = minCy; y <= maxCy; ++y) {
                for (int x = minCx; x <= maxCx; ++x) {
                     int idx = (z * resolution + y) * resolution + x;
                     grid.cells[idx].TouchCount++;
                     grid.cells[idx].AvgSizeMean += avgSize;
                     grid.cells[idx].VolRatio += ratio;
                }
            }
        }
    }
    
    // Normalize
    for (auto& cell : grid.cells) {
        if (cell.TouchCount > 0) {
            cell.AvgSizeMean /= cell.TouchCount;
            cell.VolRatio /= cell.TouchCount;
        }
    }
    std::cout << "Grid statistics generated. Resolution: " << resolution << ", WorldSize: " << worldSize << std::endl;
}

enum class DatasetMode { WKT, MESH, DT };

int main(int argc, char* argv[]) {
    std::string datasetPath = "";
    std::string outputGeometryPath = "geometry_data.txt";
    std::string outputTimingPath = "preprocessing_timing.json";
    std::string modeStr = "mesh";
    DatasetMode mode = DatasetMode::MESH;
    bool shuffle = false;
    bool generateGrid = false;
    int gridResolution = 128;
    // Default to -1.0f to indicate "auto-detect bounds" unless user overrides
    float worldSize = -1.0f;
    
    std::cout << "Arguments received:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "  [" << i << "] " << argv[i] << std::endl;
    }

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                datasetPath = argv[++i];
            }
            else if (arg == "--mode" && i + 1 < argc) {
                modeStr = argv[++i];
            }
            else if (arg == "--output-geometry" && i + 1 < argc) {
                outputGeometryPath = argv[++i];
            }
            else if (arg == "--output-timing" && i + 1 < argc) {
                outputTimingPath = argv[++i];
            }
            else if (arg == "--shuffle" || arg == "shuffle") {
                shuffle = true;
            }
            else if (arg == "--generate-grid") {
                generateGrid = true;
            }
            else if (arg == "--grid-resolution" && i + 1 < argc) {
                gridResolution = std::stoi(argv[++i]);
            }
            else if (arg == "--euler-grid-size" && i + 1 < argc) { // Alias for backward compatibility
                gridResolution = std::stoi(argv[++i]);
            }
            else if (arg == "--world-size" && i + 1 < argc) {
                worldSize = std::stof(argv[++i]);
            }
            else if (arg == "--gamma" && i + 1 < argc) {
               // ignored in preprocessing, but consume it to avoid error if passed
                ++i;
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--mode <wkt|mesh|dt>] [--dataset <path>] [--output-geometry <geometry_output_file>] [--output-timing <timing_output_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mode <wkt|mesh|dt>       Dataset loading mode (default: mesh)" << std::endl;
                std::cout << "                             wkt:  Load WKT dataset file and triangulate" << std::endl;
                std::cout << "                             mesh: Load .obj file" << std::endl;
                std::cout << "                             dt:   Load .dt file" << std::endl;
                std::cout << "  --dataset <path>           Path to dataset" << std::endl;
                std::cout << "  --output-geometry <path>   Path to text file for geometry data output" << std::endl;
                std::cout << "  --output-timing <path>     Path to JSON file for preprocessing timing output" << std::endl;
                std::cout << "  --shuffle                  Randomly translate each loaded object" << std::endl;
                std::cout << "  --generate-grid            Generate grid statistics for selectivity estimation" << std::endl;
                std::cout << "  --grid-resolution <N>      Grid resolution (default: 128)" << std::endl;
                std::cout << "  --world-size <S>           World size (default: auto-detected)" << std::endl;
                return 0;
            }
            else {
                std::cout << "Warning: Unknown argument '" << arg << "'" << std::endl;
            }
        }
    }
    
    if (modeStr == "mesh") {
        mode = DatasetMode::MESH;
    } else if (modeStr == "wkt") {
        mode = DatasetMode::WKT;
    } else if (modeStr == "dt") {
        mode = DatasetMode::DT;
    } else {
        std::cerr << "Error: Unknown mode '" << modeStr << "'. Valid modes: wkt, mesh, dt" << std::endl;
        return 1;
    }
    
    if (datasetPath.empty()) {
        std::cerr << "Error: Dataset path is required. Use --dataset <path>" << std::endl;
        return 1;
    }
    
    std::cout << "Dataset Preprocessing Tool" << std::endl;
    std::cout << "Mode: " << modeStr << std::endl;
    std::cout << "Input dataset: " << datasetPath << std::endl;
    std::cout << "Output geometry: " << outputGeometryPath << std::endl;
    std::cout << "Output timing: " << outputTimingPath << std::endl;
    
    PerformanceTimer timer;
    GeometryData geometry;
    
    if (mode == DatasetMode::WKT) {
        timer.start("Loading WKT Dataset");
        auto loader = DatasetLoaderFactory::create(DatasetType::Polygon);
        geometry = loader->load(datasetPath);
    } else {
        // For Mesh and DT modes, use auto-detection via factory
        timer.start("Loading Dataset");
        auto loader = DatasetLoaderFactory::createFromPath(datasetPath);
        if (!loader) {
            std::cerr << "Error: Could not determine loader for path: " << datasetPath << std::endl;
            return 1;
        }
        geometry = loader->load(datasetPath);
    }

    if (geometry.vertices.empty()) {
        std::cerr << "Error: Failed to load dataset or dataset is empty." << std::endl;
        return 1;
    }

    // Optional: Shuffle objects by random translation per object
    if (shuffle) {
        timer.next("Shuffling Objects (Random Translation)");
        // Build mapping from object id -> set of vertex indices used by that object
        std::unordered_map<int, std::unordered_set<unsigned int>> objectToVertexSet;
        if (geometry.indices.size() != geometry.triangleToObject.size()) {
            std::cerr << "Warning: indices and triangleToObject size mismatch; skipping shuffle." << std::endl;
        } else {
            for (size_t t = 0; t < geometry.indices.size(); ++t) {
                int objId = geometry.triangleToObject[t];
                const uint3& tri = geometry.indices[t];
                auto& vset = objectToVertexSet[objId];
                vset.insert(tri.x);
                vset.insert(tri.y);
                vset.insert(tri.z);
            }

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(-15.0f, 15.0f);

            size_t objectCount = objectToVertexSet.size();
            size_t processed = 0;
            std::cout << "Applying random translations to " << objectCount << " objects..." << std::endl;
            for (auto& kv : objectToVertexSet) {
                float tx = dist(rng);
                float ty = dist(rng);
                float tz = dist(rng);
                for (unsigned int vidx : kv.second) {
                    if (vidx < geometry.vertices.size()) {
                        geometry.vertices[vidx].x += tx;
                        geometry.vertices[vidx].y += ty;
                        geometry.vertices[vidx].z += tz;
                    }
                }
                if (++processed <= 3) {
                    std::cout << "  Object " << kv.first << " translated by (" << std::fixed << std::setprecision(2)
                              << tx << ", " << ty << ", " << tz << ")" << std::endl;
                }
            }
        }
    }

    if (generateGrid) {
        timer.next("Generating Grid Statistics");
        generateGridStats(geometry, geometry.grid, gridResolution, worldSize);
    }

    // Write geometry data
    timer.next("Writing Geometry Data");
    
    writeGeometryDataToFile(geometry, outputGeometryPath);
    
    timer.finish(outputTimingPath);
    
    std::cout << "\n=== Preprocessing Complete ===" << std::endl;
    std::cout << "Mode: " << modeStr << std::endl;
    std::cout << "Processed dataset: " << datasetPath << std::endl;
    std::cout << "Total objects (triangles): " << geometry.totalTriangles << std::endl;
    std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
    std::cout << "Total triangles: " << geometry.indices.size() << std::endl;
    std::cout << "Geometry data saved to: " << outputGeometryPath << std::endl;
    std::cout << "Timing data saved to: " << outputTimingPath << std::endl;
    
    return 0;
}
