
cmake_minimum_required(VERSION 3.18)

# Use vcpkg toolchain if available
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

project(OptiXSingleRay LANGUAGES CXX CUDA)

cmake_policy(SET CMP0167 NEW)

# Ensure all targets use the DLL version of the MSVC runtime (avoids CRT conflicts)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add preprocessor definitions to prevent Windows min/max macro conflicts
if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# Find required packages (prefer vcpkg if available)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CGAL REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)
include(cmake/FindOptiX.cmake)

include(cmake/nvcuda_compile_module.cmake)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUDA_SOURCES ${SRC_DIR}/raytracing.cu)
set(RESULT_COMPACTION_CUDA ${SRC_DIR}/result_compaction.cu)
set(CPP_SOURCE   ${SRC_DIR}/rayspace.cpp)
set(FILTER_REFINE_SOURCE ${SRC_DIR}/rayspace_filter_refine.cpp)
## Removed legacy dataset_loader.cpp
set(TRIANGULATE_SOURCE ${SRC_DIR}/triangulate_dataset.cpp)
set(TRIANGULATION_LIB_SOURCE ${SRC_DIR}/triangulation.cpp)
set(TIMER_SOURCE ${SRC_DIR}/timer.cpp)
set(TRIANGULATE_SINGLE_SOURCE ${SRC_DIR}/triangulate_single_polygon.cpp)
set(PREPROCESS_SOURCE ${SRC_DIR}/preprocess_dataset.cpp)

# New dataset loader classes
set(DATASET_DIR ${SRC_DIR}/dataset)
set(DATASET_COMMON_DIR ${DATASET_DIR}/common)
set(DATASET_PREPROCESS_DIR ${DATASET_DIR}/preprocess)
set(DATASET_RUNTIME_DIR ${DATASET_DIR}/runtime)

set(DATASET_SOURCES
    ${DATASET_COMMON_DIR}/DatasetUtils.cpp
    ${DATASET_PREPROCESS_DIR}/DatasetLoaderFactory.cpp
    ${DATASET_PREPROCESS_DIR}/PolygonDatasetLoader.cpp
    ${DATASET_PREPROCESS_DIR}/MeshDatasetLoader.cpp
    ${DATASET_RUNTIME_DIR}/GeometryIO.cpp
    ${DATASET_RUNTIME_DIR}/PointIO.cpp)

set(PTX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

NVCUDA_COMPILE_MODULE(
    SOURCES       ${CUDA_SOURCES}
    TARGET_PATH   ${PTX_OUTPUT_DIR}
    EXTENSION     "${OPTIX_MODULE_EXTENSION}"
    GENERATED_FILES PTX_FILES
    NVCC_OPTIONS  "${OPTIX_PROGRAM_TARGET}" "--gpu-architecture=sm_75" "--relocatable-device-code=true" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}"
)

add_executable(raytracer ${CPP_SOURCE} ${RESULT_COMPACTION_CUDA} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

add_executable(raytracer_filter_refine ${FILTER_REFINE_SOURCE} ${RESULT_COMPACTION_CUDA} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

add_executable(triangulate_single_polygon ${TRIANGULATE_SINGLE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

add_executable(preprocess_dataset ${PREPROCESS_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

# Platform-specific include directories
target_include_directories(raytracer PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${OptiX_INCLUDE} 
    ${SRC_DIR}
    ${DATASET_COMMON_DIR}
    ${DATASET_RUNTIME_DIR})

target_include_directories(raytracer_filter_refine PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${OptiX_INCLUDE} 
    ${SRC_DIR}
    ${DATASET_COMMON_DIR}
    ${DATASET_RUNTIME_DIR})

target_include_directories(triangulate_single_polygon PRIVATE 
    ${SRC_DIR}
    ${DATASET_COMMON_DIR})
    
target_include_directories(preprocess_dataset PRIVATE 
    ${SRC_DIR}
    ${DATASET_COMMON_DIR}
    ${DATASET_PREPROCESS_DIR})

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)
add_dependencies(raytracer_filter_refine copy_ptx)


# OptiX host functions are loaded through the stub headers at runtime, so linking the explicit
# OptiX libraries is unnecessary and breaks the build when those import libraries are not found.
# RayJoin works without that explicit linkage, so mirror the same approach here.
target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)
target_link_libraries(raytracer_filter_refine PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)

# Define INCLUDE_OPTIX for raytracer only
target_compile_definitions(raytracer PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_filter_refine PRIVATE INCLUDE_OPTIX)

# Linking for triangulate_single_polygon and preprocess_dataset
target_link_libraries(triangulate_single_polygon PRIVATE CGAL::CGAL Boost::filesystem Boost::system)
target_link_libraries(preprocess_dataset PRIVATE CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_filter_refine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(triangulate_single_polygon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(preprocess_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

# add_executable(triangulate_dataset ${TRIANGULATE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

# target_include_directories(triangulate_dataset PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/CDT/CDT/include
#     C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

# set_target_properties(triangulate_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 