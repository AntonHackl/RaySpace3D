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
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <set>
#include <vector>
// #include "Geometry.h"
#include "../../common/include/Geometry.h"
#include "DatasetLoaderFactory.h"
#include "timer.h"

// Helper for float3 operations for Euler Histogram
inline float3 f3_min(const float3& a, const float3& b) {
    return { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
}
inline float3 f3_max(const float3& a, const float3& b) {
    return { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
}

EulerHistogram computeEulerHistogram(const GeometryData& geometry, int nx, int ny, int nz) {
    EulerHistogram hist;
    hist.nx = nx; hist.ny = ny; hist.nz = nz;
    size_t totalCells = (size_t)nx * ny * nz;
    hist.v_counts.resize(totalCells, 0);
    hist.e_counts.resize(totalCells, 0);
    hist.f_counts.resize(totalCells, 0);

    // 1. Compute Bounds
    hist.minBound = { FLT_MAX, FLT_MAX, FLT_MAX };
    hist.maxBound = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

    if (geometry.vertices.empty()) return hist;

    for (const auto& v : geometry.vertices) {
        hist.minBound = f3_min(hist.minBound, v);
        hist.maxBound = f3_max(hist.maxBound, v);
    }

    // Add small epsilon padding to maxBound to ensure points on the boundary fall inside the last cell
    const float epsilon = 1e-4f;
    hist.maxBound.x += epsilon;
    hist.maxBound.y += epsilon;
    hist.maxBound.z += epsilon;

    hist.cellSize.x = (hist.maxBound.x - hist.minBound.x) / nx;
    hist.cellSize.y = (hist.maxBound.y - hist.minBound.y) / ny;
    hist.cellSize.z = (hist.maxBound.z - hist.minBound.z) / nz;

    // Helper lambda to get cell index
    auto getCellIndex = [&](const float3& p) -> int {
        int ix = static_cast<int>((p.x - hist.minBound.x) / hist.cellSize.x);
        int iy = static_cast<int>((p.y - hist.minBound.y) / hist.cellSize.y);
        int iz = static_cast<int>((p.z - hist.minBound.z) / hist.cellSize.z);

        // Clamp to be safe
        ix = std::max(0, std::min(ix, nx - 1));
        iy = std::max(0, std::min(iy, ny - 1));
        iz = std::max(0, std::min(iz, nz - 1));

        return iz * (nx * ny) + iy * nx + ix;
    };

    // 2. Bin Vertices
    for (const auto& v : geometry.vertices) {
        hist.v_counts[getCellIndex(v)]++;
    }

    // 3. Bin Unique Edges
    // Extract unique edges first: (u, v) with u < v
    std::vector<std::pair<unsigned int, unsigned int>> edges;
    edges.reserve(geometry.indices.size() * 3);

    for (const auto& tri : geometry.indices) {
        unsigned int vIdx[3] = { tri.x, tri.y, tri.z };
        for (int i = 0; i < 3; ++i) {
            unsigned int a = vIdx[i];
            unsigned int b = vIdx[(i + 1) % 3];
            if (a > b) std::swap(a, b);
            edges.push_back({ a, b });
        }
    }

    // Deduplicate edges
    std::sort(edges.begin(), edges.end());
    auto last = std::unique(edges.begin(), edges.end());
    edges.erase(last, edges.end());

    // Bin edges by midpoint
    for (const auto& e : edges) {
        if (e.first >= geometry.vertices.size() || e.second >= geometry.vertices.size()) continue;

        const float3& p1 = geometry.vertices[e.first];
        const float3& p2 = geometry.vertices[e.second];
        float3 midpoint = {
            (p1.x + p2.x) * 0.5f,
            (p1.y + p2.y) * 0.5f,
            (p1.z + p2.z) * 0.5f
        };
        hist.e_counts[getCellIndex(midpoint)]++;
    }

    // 4. Bin Faces (Triangles)
    // Bin by centroid
    // Also bin Objects: we need to track unique objects per cell
    // Since we can have many objects per cell, we use an array of sets for the grid
    std::vector<std::set<int>> cellObjects(totalCells);

    for (size_t t = 0; t < geometry.indices.size(); ++t) {
        const auto& tri = geometry.indices[t];
        if (tri.x >= geometry.vertices.size() || tri.y >= geometry.vertices.size() || tri.z >= geometry.vertices.size()) continue;

        const float3& p1 = geometry.vertices[tri.x];
        const float3& p2 = geometry.vertices[tri.y];
        const float3& p3 = geometry.vertices[tri.z];
        float3 centroid = {
            (p1.x + p2.x + p3.x) / 3.0f,
            (p1.y + p2.y + p3.y) / 3.0f,
            (p1.z + p2.z + p3.z) / 3.0f
        };
        
        int cellIdx = getCellIndex(centroid);
        hist.f_counts[cellIdx]++;
        
        if (t < geometry.triangleToObject.size()) {
            cellObjects[cellIdx].insert(geometry.triangleToObject[t]);
        }
    }

    // Convert sets to counts
    hist.object_counts.resize(totalCells);
    for (size_t i = 0; i < totalCells; ++i) {
        hist.object_counts[i] = static_cast<int>(cellObjects[i].size());
    }

    return hist;
}

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

    // Append Euler Histogram
    const auto& hist = geometry.eulerHistogram;
    if (hist.nx > 0) {
        oss << "euler_grid_dims: " << hist.nx << ' ' << hist.ny << ' ' << hist.nz << '\n';
        oss << "euler_bbox: " << hist.minBound.x << ' ' << hist.minBound.y << ' ' << hist.minBound.z
            << ' ' << hist.maxBound.x << ' ' << hist.maxBound.y << ' ' << hist.maxBound.z << '\n';

        oss << "euler_data_v: ";
        for (size_t i = 0; i < hist.v_counts.size(); ++i) {
            oss << hist.v_counts[i] << (i + 1 < hist.v_counts.size() ? " " : "");
        }
        oss << '\n';

        oss << "euler_data_e: ";
        for (size_t i = 0; i < hist.e_counts.size(); ++i) {
            oss << hist.e_counts[i] << (i + 1 < hist.e_counts.size() ? " " : "");
        }
        oss << '\n';

        oss << "euler_data_f: ";
        for (size_t i = 0; i < hist.f_counts.size(); ++i) {
            oss << hist.f_counts[i] << (i + 1 < hist.f_counts.size() ? " " : "");
        }
        oss << '\n';

        oss << "euler_data_o: ";
        for (size_t i = 0; i < hist.object_counts.size(); ++i) {
            oss << hist.object_counts[i] << (i + 1 < hist.object_counts.size() ? " " : "");
        }
        oss << '\n';
    }

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
    int eulerGridSize = 0; // Default 0 means no histogram unless specified

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
            else if (arg == "--euler-grid-size" && i + 1 < argc) {
                eulerGridSize = std::stoi(argv[++i]);
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--mode <wkt|mesh>] [--dataset <path>] [--output-geometry <geometry_output_file>] [--output-timing <timing_output_file>]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --mode <wkt|mesh>          Dataset loading mode (default: wkt)" << std::endl;
                std::cout << "                             wkt:  Load WKT dataset file and triangulate" << std::endl;
                std::cout << "                             mesh: Load single .obj file with multiple objects (indexed by 'o' statements)" << std::endl;
                std::cout << "  --dataset <path>           Path to dataset (WKT file for wkt mode, .obj file for mesh mode)" << std::endl;
                std::cout << "  --output-geometry <path>   Path to text file for geometry data output (default: geometry_data.txt)" << std::endl;
                std::cout << "  --output-timing <path>     Path to JSON file for preprocessing timing output (default: preprocessing_timing.json)" << std::endl;
                std::cout << "  --shuffle                  Randomly translate each loaded object by a vector in [-100, 100] on x/y/z" << std::endl;
                std::cout << "  --euler-grid-size <N>      Generate Euler Histogram with grid size N^3 (default: 0, disabled)" << std::endl;
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
        // Use createFromPath to auto-detect .obj vs .dt files
        auto loader = DatasetLoaderFactory::createFromPath(datasetPath);
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

    // Compute Euler Histogram if requested
    if (eulerGridSize > 0) {
        timer.next("Computing Euler Histogram");
        std::cout << "Generating Euler Histogram with grid size " << eulerGridSize << "^3..." << std::endl;
        geometry.eulerHistogram = computeEulerHistogram(geometry, eulerGridSize, eulerGridSize, eulerGridSize);
        std::cout << "Euler Histogram computed." << std::endl;
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
