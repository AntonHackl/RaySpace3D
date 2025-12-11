
cmake_minimum_required(VERSION 3.18)

project(OptiXSingleRay LANGUAGES CXX CUDA)

cmake_policy(SET CMP0167 NEW)

# Ensure all targets use the DLL version of the MSVC runtime (avoids CRT conflicts)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position-independent code for CUDA to work with conda compilers
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIE")
    # Add library search paths for system libraries (Linux only)
    link_directories(/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
endif()

# Export compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add preprocessor definitions to prevent Windows min/max macro conflicts
if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# Find required packages from conda environment
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CGAL REQUIRED)

# TinyObjLoader - handle both conda and manual installation
find_package(tinyobjloader CONFIG QUIET)
if(NOT tinyobjloader_FOUND)
    # Check for local tinyobjloader
    set(TINYOBJLOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader)
    if(EXISTS ${TINYOBJLOADER_DIR}/tiny_obj_loader.h AND EXISTS ${TINYOBJLOADER_DIR}/tiny_obj_loader.cc)
        # Build tinyobjloader as a static library
        add_library(tinyobjloader STATIC ${TINYOBJLOADER_DIR}/tiny_obj_loader.cc)
        target_include_directories(tinyobjloader PUBLIC ${TINYOBJLOADER_DIR})
        add_library(tinyobjloader::tinyobjloader ALIAS tinyobjloader)
        message(STATUS "Building local TinyObjLoader from: ${TINYOBJLOADER_DIR}")
    else()
        message(FATAL_ERROR "TinyObjLoader not found. Please download it to external/tinyobjloader/")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Detect conda environment and configure CUDA paths
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: ${CONDA_PREFIX}")
    
    # Add conda prefix to CMAKE_PREFIX_PATH for library discovery
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
    
    # On Windows, conda CUDA is typically in Library/bin
    # On Linux, it's in bin
    if(WIN32)
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/Library/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/Library/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/Library/lib")
    else()
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/lib")
    endif()
    
    # Check if conda CUDA nvcc exists
    if(EXISTS "${CONDA_CUDA_BIN}/nvcc${CMAKE_EXECUTABLE_SUFFIX}" OR EXISTS "${CONDA_CUDA_BIN}/nvcc")
        message(STATUS "Found conda CUDA toolkit at: ${CONDA_PREFIX}")
        # Set CUDA paths explicitly for conda CUDA (before find_package)
        if(NOT DEFINED CUDA_PATH)
            set(CUDA_PATH "${CONDA_PREFIX}" CACHE PATH "CUDA installation path")
        endif()
        if(NOT DEFINED CMAKE_CUDA_COMPILER)
            if(EXISTS "${CONDA_CUDA_BIN}/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
                set(CMAKE_CUDA_COMPILER "${CONDA_CUDA_BIN}/nvcc${CMAKE_EXECUTABLE_SUFFIX}" CACHE FILEPATH "CUDA compiler")
            else()
                set(CMAKE_CUDA_COMPILER "${CONDA_CUDA_BIN}/nvcc" CACHE FILEPATH "CUDA compiler")
            endif()
        endif()
        # Add conda CUDA include and library paths
        include_directories("${CONDA_CUDA_INCLUDE}")
        link_directories("${CONDA_CUDA_LIB}")
    else()
        if(WIN32)
            message(FATAL_ERROR "CONDA_PREFIX is set but conda CUDA toolkit not found. On Windows, conda CUDA is required. Please install: conda install -c nvidia cuda-toolkit=12.8")
        else()
            message(WARNING "CONDA_PREFIX is set but nvcc not found in conda environment. Falling back to system CUDA.")
        endif()
    endif()
endif()

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
    ${CUDA_INCLUDE_DIRS} 
    ${OptiX_INCLUDE}
    ${SRC_DIR}
    ${DATASET_COMMON_DIR}
    ${DATASET_PREPROCESS_DIR})

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)
add_dependencies(raytracer_filter_refine copy_ptx)

# Copy PTX file to bin directory so executables can find it
foreach(TARGET_NAME raytracer raytracer_filter_refine)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PTX_FILE}"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/raytracing${OPTIX_MODULE_EXTENSION}"
        COMMENT "Copying PTX file to ${TARGET_NAME} output directory"
    )
endforeach()


# OptiX host functions are loaded through the stub headers at runtime, so linking the explicit
# OptiX libraries is unnecessary and breaks the build when those import libraries are not found.
# RayJoin works without that explicit linkage, so mirror the same approach here.
target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)
target_link_libraries(raytracer_filter_refine PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)
target_link_libraries(preprocess_dataset PRIVATE CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system ${CUDA_LIBRARIES})

# Define INCLUDE_OPTIX for raytracer only
target_compile_definitions(raytracer PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_filter_refine PRIVATE INCLUDE_OPTIX)

# Linking for triangulate_single_polygon
target_link_libraries(triangulate_single_polygon PRIVATE CGAL::CGAL Boost::filesystem Boost::system)

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_filter_refine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(triangulate_single_polygon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(preprocess_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

# add_executable(triangulate_dataset ${TRIANGULATE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

# target_include_directories(triangulate_dataset PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/CDT/CDT/include
#     C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

# set_target_properties(triangulate_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 