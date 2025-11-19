cmake_minimum_required(VERSION 3.18)

# Only use vcpkg on Windows
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/anton/Documents/Uni/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

project(OptiXSingleRay LANGUAGES CXX CUDA)

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

# Find required packages
if(WIN32)
    # Windows: use vcpkg packages
    find_package(Boost REQUIRED COMPONENTS geometry)
    find_package(CGAL REQUIRED)
    find_package(tinyobjloader CONFIG REQUIRED)
else()
    # Linux: use system packages
    set(AE_DEPS_DIR "/root/media/PPoPPAE/deps")
    list(APPEND CMAKE_PREFIX_PATH "${AE_DEPS_DIR}")
    find_package(PkgConfig REQUIRED)
    find_package(Boost REQUIRED COMPONENTS system filesystem)
    find_package(CGAL REQUIRED)
    find_package(tinyobjloader CONFIG REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)
include(cmake/FindOptiX.cmake)

include(cmake/nvcuda_compile_module.cmake)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUDA_SOURCES ${SRC_DIR}/raytracing.cu)
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

add_executable(raytracer ${CPP_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

add_executable(raytracer_filter_refine ${FILTER_REFINE_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

add_executable(triangulate_single_polygon ${TRIANGULATE_SINGLE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

add_executable(preprocess_dataset ${PREPROCESS_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

# Platform-specific include directories
if(WIN32)
    target_include_directories(raytracer PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_RUNTIME_DIR}
        C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

    target_include_directories(raytracer_filter_refine PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_RUNTIME_DIR}
        C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

    target_include_directories(triangulate_single_polygon PRIVATE 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)
        
    target_include_directories(preprocess_dataset PRIVATE 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_PREPROCESS_DIR}
        C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)
else()
    target_include_directories(raytracer PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_RUNTIME_DIR}
        ${Boost_INCLUDE_DIRS})

    target_include_directories(raytracer_filter_refine PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_RUNTIME_DIR}
        ${Boost_INCLUDE_DIRS})

    target_include_directories(triangulate_single_polygon PRIVATE 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${Boost_INCLUDE_DIRS})
        
    target_include_directories(preprocess_dataset PRIVATE 
        ${SRC_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_PREPROCESS_DIR}
        ${Boost_INCLUDE_DIRS})
endif()

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)
add_dependencies(raytracer_filter_refine copy_ptx)

# OptiX host functions are loaded through the stub headers at runtime, so linking the explicit
# OptiX libraries is unnecessary and breaks the build when those import libraries are not found.
# RayJoin works without that explicit linkage, so mirror the same approach here.
target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} CGAL::CGAL tinyobjloader::tinyobjloader)
target_link_libraries(raytracer_filter_refine PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} CGAL::CGAL tinyobjloader::tinyobjloader)

# Define INCLUDE_OPTIX for raytracer only
target_compile_definitions(raytracer PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_filter_refine PRIVATE INCLUDE_OPTIX)

# Platform-specific linking for triangulate_single_polygon and preprocess_dataset
if(WIN32)
    target_link_libraries(triangulate_single_polygon PRIVATE CGAL::CGAL)
    target_link_libraries(preprocess_dataset PRIVATE CGAL::CGAL tinyobjloader::tinyobjloader)
else()
    # Link Boost libraries on Linux
    target_link_libraries(triangulate_single_polygon PRIVATE ${Boost_LIBRARIES} CGAL::CGAL)
    target_link_libraries(preprocess_dataset PRIVATE ${Boost_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader)
endif()

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_filter_refine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(triangulate_single_polygon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(preprocess_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

# add_executable(triangulate_dataset ${TRIANGULATE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

# target_include_directories(triangulate_dataset PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/CDT/CDT/include
#     C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

# set_target_properties(triangulate_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 