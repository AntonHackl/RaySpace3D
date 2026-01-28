#include "GeometryIO.h"
#include "../../../common/include/BinaryIO.h"

#include <iostream>

GeometryData loadGeometryFromFile(const std::string& geometryFilePath) {
    std::cout << "=== Loading Preprocessed Geometry (Binary) ===" << std::endl;
    return RaySpace::IO::readBinaryFile(geometryFilePath);
}
