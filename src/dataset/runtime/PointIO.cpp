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
    std::string line; int lineNum = 0;
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
                        pointData.positions.push_back({x, y, z});
                    } else {
                        ss.clear();
                        ss.str(coords);
                        if (ss >> x >> y) {
                            pointData.positions.push_back({x, y, -1.0f});
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
                    pointData.positions.push_back({x, y, z});
                }
            } catch (const std::exception& e) {
                (void)e; // ignore
                continue;
            }
        }
    }
    pointData.numPoints = pointData.positions.size();
    std::cout << "Loaded " << pointData.numPoints << " points from dataset" << std::endl;
    std::cout << "=============================\n" << std::endl;
    return pointData;
}
