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
#include "dataset/common/Geometry.h"
#include "dataset/preprocess/DatasetLoaderFactory.h"
#include "timer.h"

void writeGeometryDataToFile(const GeometryData& geometry, const std::string& filename) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);

    oss << "vertices: ";
    for (size_t i = 0; i < geometry.vertices.size(); ++i) {
        const auto& v = geometry.vertices[i];
        oss << v.x << ' ' << v.y << ' ' << v.z;
        if (i + 1 < geometry.vertices.size()) oss << ' ';
    }
    oss << '\n';

    oss << "indices: ";
    for (size_t i = 0; i < geometry.indices.size(); ++i) {
        const auto& tri = geometry.indices[i];
        oss << tri.x << ' ' << tri.y << ' ' << tri.z;
        if (i + 1 < geometry.indices.size()) oss << ' ';
    }
    oss << '\n';

    oss << "triangleToObject: ";
    for (size_t i = 0; i < geometry.triangleToObject.size(); ++i) {
        oss << geometry.triangleToObject[i];
        if (i + 1 < geometry.triangleToObject.size()) oss << ' ';
    }
    oss << '\n';

    oss << "total_triangles: " << geometry.totalTriangles << '\n';

    const std::string buffer = std::move(oss).str();
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();

    std::cout << "Geometry data written to: " << filename << std::endl;
}

enum class DatasetMode { WKT, MESH };

int main(int argc, char* argv[]) {
    std::string datasetPath = "";
    std::string outputGeometryPath = "geometry_data.txt";
    std::string outputTimingPath = "preprocessing_timing.json";
    std::string modeStr = "mesh";
    DatasetMode mode = DatasetMode::WKT;
    bool shuffle = false;
    
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
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--mode <wkt|mesh>] [--dataset <path>] [--output-geometry <geometry_output_file>] [--output-timing <timing_output_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mode <wkt|mesh>          Dataset loading mode (default: wkt)" << std::endl;
                std::cout << "                             wkt:  Load WKT dataset file and triangulate" << std::endl;
                std::cout << "                             mesh: Load .obj files from directory recursively" << std::endl;
                std::cout << "  --dataset <path>           Path to dataset (WKT file for wkt mode, directory for mesh mode)" << std::endl;
                std::cout << "  --output-geometry <path>   Path to text file for geometry data output (default: geometry_data.txt)" << std::endl;
                std::cout << "  --output-timing <path>     Path to JSON file for preprocessing timing output (default: preprocessing_timing.json)" << std::endl;
                std::cout << "  --shuffle                  Randomly translate each loaded object by a vector in [-100, 100] on x/y/z" << std::endl;
                std::cout << "  --help, -h                 Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    if (modeStr == "mesh") {
        mode = DatasetMode::MESH;
    } else if (modeStr == "wkt") {
        mode = DatasetMode::WKT;
    } else {
        std::cerr << "Error: Unknown mode '" << modeStr << "'. Valid modes: wkt, mesh" << std::endl;
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
        if (geometry.vertices.empty()) {
            std::cerr << "Error: Failed to load/triangulate WKT dataset." << std::endl;
            return 1;
        }
    } else if (mode == DatasetMode::MESH) {
        timer.start("Loading Mesh Dataset");
        auto loader = DatasetLoaderFactory::create(DatasetType::Mesh);
        geometry = loader->load(datasetPath);
        if (geometry.vertices.empty()) {
            std::cerr << "Error: Failed to load mesh dataset." << std::endl;
            return 1;
        }
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
