#include "PointIO.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

PointData loadPointDataset(const std::string& pointDatasetPath) {
    PointData pointData;
    if (pointDatasetPath.empty()) {
        std::cout << "No point dataset provided, using default test points" << std::endl;
        pointData.positions = { {0.0f,0.0f,-1.0f},{0.5f,0.5f,-1.0f},{1.0f,1.0f,-1.0f} };
        pointData.numPoints = 3; return pointData; }

    std::cout << "=== Loading Point Dataset ===" << std::endl;
    std::cout << "Loading points from: " << pointDatasetPath << std::endl;

    FILE* file = fopen(pointDatasetPath.c_str(), "r");
    if (!file) { 
        std::cerr << "Error: Could not open point dataset file: " << pointDatasetPath << std::endl; 
        return pointData; 
    }

    // Direct loading into pinned memory vector
    // This avoids double allocation and copying
    pointData.positions.reserve(1000000); 

    char line[1024]; // Buffer for line reading
    while (fgets(line, sizeof(line), file)) {
        char* ptr = line;
        while (*ptr == ' ' || *ptr == '\t') ptr++;

        if (*ptr == '\0' || *ptr == '\r' || *ptr == '\n' || *ptr == '#') continue;

        // Optimized parsing for "POINT Z (<x> <y> <z>)"
        // Manual parsing is significantly faster than regex or stringstream
        if (strncmp(ptr, "POINT", 5) == 0) {
            char* start = strchr(ptr, '(');
            if (start) {
                start++;
                char* endPtr;
                float x = strtof(start, &endPtr);
                float y = strtof(endPtr, &endPtr);
                float z = strtof(endPtr, &endPtr);
                pointData.positions.push_back({x, y, z});
            }
        } 
        // Fallback for simple list format (e.g. "1.0 2.0 3.0")
        else if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-') {
            char* endPtr;
            float x = strtof(ptr, &endPtr);
            float y = strtof(endPtr, &endPtr);
            float z = strtof(endPtr, &endPtr);
            if (ptr != endPtr) {
                pointData.positions.push_back({x, y, z});
            }
        }
    }
    fclose(file);
    
    pointData.numPoints = pointData.positions.size();
    std::cout << "Found " << pointData.numPoints << " points." << std::endl;

    std::cout << "Loaded " << pointData.numPoints << " points directly into pinned memory (std::vector)" << std::endl;
    std::cout << "Ready for GPU transfer (zero-copy)" << std::endl;
    std::cout << "=============================\n" << std::endl;
    return pointData;
}
