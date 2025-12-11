#include "PointIO.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

PointData loadPointDataset(const std::string& pointDatasetPath) {
    PointData pointData;
    if (pointDatasetPath.empty()) {
        std::cout << "No point dataset provided, using default test points" << std::endl;
        pointData.positions = { {0.0f,0.0f,-1.0f},{0.5f,0.5f,-1.0f},{1.0f,1.0f,-1.0f} };
        pointData.numPoints = 3; return pointData; }
    std::cout << "=== Loading Point Dataset ===" << std::endl;
    std::cout << "Loading points from: " << pointDatasetPath << std::endl;
    std::ifstream file(pointDatasetPath);
    if (!file.is_open()) { std::cerr << "Error: Could not open point dataset file: " << pointDatasetPath << std::endl; return pointData; }
    
    // First pass: count points to allocate pinned memory once
    std::string line; int lineNum = 0;
    size_t pointCount = 0;
    std::streampos fileStart = file.tellg();
    
    std::cout << "Counting points..." << std::endl;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0]=='#') continue;
        if (line.find("POINT") != std::string::npos || (!line.empty() && std::isdigit(line[0]) || line[0] == '-')) {
            pointCount++;
        }
    }
    
    std::cout << "Found " << pointCount << " points. Allocating pinned memory..." << std::endl;
    pointData.pinnedBuffers.allocate(pointCount);
    float3* positions_pinned = pointData.pinnedBuffers.positions_pinned;
    
    // Second pass: load directly into pinned memory
    file.clear();
    file.seekg(fileStart);
    size_t idx = 0;
    lineNum = 0;
    
    while (std::getline(file, line)) {
        lineNum++;
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0]=='#') continue;
        if (line.find("POINT") != std::string::npos) {
            try {
                size_t start = line.find('('); size_t end = line.find(')');
                if (start!=std::string::npos && end!=std::string::npos) {
                    std::string coords = line.substr(start+1, end-start-1);
                    std::stringstream ss(coords);
                    float x, y, z;
                    if (ss >> x >> y >> z) {
                        positions_pinned[idx++] = {x, y, z};
                    } else {
                        ss.clear();
                        ss.str(coords);
                        if (ss >> x >> y) {
                            positions_pinned[idx++] = {x, y, -1.0f};
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not parse point on line " << lineNum << ": " << e.what() << std::endl; continue;
            }
        } else {
            try {
                std::stringstream ss(line);
                float x, y, z;
                if (ss >> x >> y >> z) {
                    positions_pinned[idx++] = {x, y, z};
                }
            } catch (const std::exception& e) {
                (void)e; // ignore
                continue;
            }
        }
    }
    pointData.numPoints = idx;
    std::cout << "Loaded " << pointData.numPoints << " points directly into pinned memory" << std::endl;
    std::cout << "Ready for GPU transfer (zero-copy)" << std::endl;
    std::cout << "=============================\n" << std::endl;
    return pointData;
}
