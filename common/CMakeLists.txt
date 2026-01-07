cmake_minimum_required(VERSION 3.18)
project(RaySpace3DCommon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common source files
set(COMMON_SOURCES
    src/timer.cpp
    src/triangulation.cpp
    include/DatasetUtils.cpp
)

# Create a static library
add_library(rayspace3d_common STATIC ${COMMON_SOURCES})

# Public include directories
target_include_directories(rayspace3d_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Find CGAL (for triangulation)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

target_link_libraries(rayspace3d_common PUBLIC
    CGAL::CGAL
    Boost::filesystem
    Boost::system
)
