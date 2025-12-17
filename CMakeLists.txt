cmake_minimum_required(VERSION 3.18)

# Force system compiler on Linux to avoid conda gcc/glibc conflicts
# Must be set BEFORE project() command
if(UNIX AND NOT APPLE AND NOT WIN32)
    if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED CMAKE_CXX_COMPILER)
        find_program(SYSTEM_CXX_COMPILER NAMES g++ PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(SYSTEM_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${SYSTEM_CXX_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
            message(STATUS "Forcing system C++ compiler: ${SYSTEM_CXX_COMPILER}")
        endif()
    endif()
endif()

project(OptiXSingleRay LANGUAGES CXX CUDA)

# Policy to correctly handle MSVC Runtime Library flags (Static vs Dynamic)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0167 NEW)

# -----------------------------------------------------------------------
# 1. Global Settings & Runtime Configuration
# -----------------------------------------------------------------------

if(MSVC)
    # Use MultiThreaded Debug DLL (/MDd) for Debug builds to match Conda/Boost
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIE")
    link_directories(/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
    
    # Ensure system linker finds standard libraries correctly
    # Add --sysroot to avoid conda gcc sysroot issues
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# -----------------------------------------------------------------------
# 2. Dependency Finding
# -----------------------------------------------------------------------

# Detect conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: ${CONDA_PREFIX}")
    
    # Add conda paths to search paths with highest priority
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
    
    if(WIN32)
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/Library/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/Library/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/Library/lib")
    else()
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/lib")
        
        # Ensure conda library paths are included and boost can be found
        include_directories(SYSTEM "${CONDA_PREFIX}/include")
        link_directories("${CONDA_PREFIX}/lib")
    endif()
    
    # Configure CUDA compiler from Conda if available
    if(EXISTS "${CONDA_CUDA_BIN}/nvcc${CMAKE_EXECUTABLE_SUFFIX}" OR EXISTS "${CONDA_CUDA_BIN}/nvcc")
        if(NOT DEFINED CUDA_PATH)
            set(CUDA_PATH "${CONDA_PREFIX}" CACHE PATH "CUDA installation path")
        endif()
        if(NOT DEFINED CMAKE_CUDA_COMPILER)
            file(GLOB NVCC_EXEC "${CONDA_CUDA_BIN}/nvcc*")
            list(GET NVCC_EXEC 0 NVCC_PATH)
            set(CMAKE_CUDA_COMPILER "${NVCC_PATH}" CACHE FILEPATH "CUDA compiler")
        endif()
        include_directories("${CONDA_CUDA_INCLUDE}")
        link_directories("${CONDA_CUDA_LIB}")
    endif()
endif()

find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CGAL REQUIRED)
find_package(CUDA REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(cmake/FindOptiX.cmake)
include(cmake/nvcuda_compile_module.cmake)

# TinyObjLoader
find_package(tinyobjloader CONFIG QUIET)
if(NOT tinyobjloader_FOUND)
    set(TINYOBJLOADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader)
    if(EXISTS ${TINYOBJLOADER_DIR}/tiny_obj_loader.h AND EXISTS ${TINYOBJLOADER_DIR}/tiny_obj_loader.cc)
        add_library(tinyobjloader STATIC ${TINYOBJLOADER_DIR}/tiny_obj_loader.cc)
        target_include_directories(tinyobjloader PUBLIC ${TINYOBJLOADER_DIR})
        add_library(tinyobjloader::tinyobjloader ALIAS tinyobjloader)
        
        # Ensure static lib uses the same Runtime as the App (DLL)
        if(MSVC)
            set_property(TARGET tinyobjloader PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        endif()
    else()
        message(FATAL_ERROR "TinyObjLoader not found. Please download it to external/tinyobjloader/")
    endif()
endif()

# -----------------------------------------------------------------------
# 3. Source Definitions
# -----------------------------------------------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# CUDA Sources
set(CUDA_SOURCES ${SRC_DIR}/raytracing.cu)
set(RESULT_COMPACTION_SOURCE ${SRC_DIR}/result_compaction.cu)

# C++ Application Sources
set(CPP_SOURCE    ${SRC_DIR}/applications/rayspace_main.cpp)
set(FILTER_REFINE_SOURCE ${SRC_DIR}/applications/rayspace_filter_refine_main.cpp)
set(TRIANGULATE_SINGLE_SOURCE ${SRC_DIR}/triangulate_single_polygon.cpp)
set(PREPROCESS_SOURCE ${SRC_DIR}/preprocess_dataset.cpp)

# Helper Sources
set(TRIANGULATION_LIB_SOURCE ${SRC_DIR}/triangulation.cpp)
set(TIMER_SOURCE ${SRC_DIR}/timer.cpp)

# Modules
set(OPTIX_DIR ${SRC_DIR}/optix)
set(GEOMETRY_DIR ${SRC_DIR}/geometry)
set(RAYTRACING_DIR ${SRC_DIR}/raytracing)

set(OPTIX_SOURCES
    ${OPTIX_DIR}/OptixContext.cpp
    ${OPTIX_DIR}/OptixPipeline.cpp
    ${OPTIX_DIR}/OptixAccelerationStructure.cpp)

set(GEOMETRY_SOURCES
    ${GEOMETRY_DIR}/BoundingBox.cpp
    ${GEOMETRY_DIR}/GeometryUploader.cpp)

set(RAYTRACING_SOURCES
    ${RAYTRACING_DIR}/RayLauncher.cpp
    ${RAYTRACING_DIR}/ResultProcessor.cpp
    ${RAYTRACING_DIR}/FilterRefine.cpp)

# Dataset Modules
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

# -----------------------------------------------------------------------
# 4. PTX Compilation
# -----------------------------------------------------------------------

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

# -----------------------------------------------------------------------
# 5. Targets
# -----------------------------------------------------------------------

# Added RESULT_COMPACTION_SOURCE to include GPU hit counting/compaction kernels
add_executable(raytracer ${CPP_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES} ${OPTIX_SOURCES} ${GEOMETRY_SOURCES} ${RAYTRACING_SOURCES} ${RESULT_COMPACTION_SOURCE})
add_executable(raytracer_filter_refine ${FILTER_REFINE_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES} ${OPTIX_SOURCES} ${GEOMETRY_SOURCES} ${RAYTRACING_SOURCES} ${RESULT_COMPACTION_SOURCE})
add_executable(triangulate_single_polygon ${TRIANGULATE_SINGLE_SOURCE} ${TRIANGULATION_LIB_SOURCE})
add_executable(preprocess_dataset ${PREPROCESS_SOURCE} ${TRIANGULATION_LIB_SOURCE} ${TIMER_SOURCE} ${DATASET_SOURCES})

foreach(TARGET_NAME raytracer raytracer_filter_refine preprocess_dataset)
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${OPTIX_DIR}
        ${GEOMETRY_DIR}
        ${RAYTRACING_DIR}
        ${DATASET_COMMON_DIR}
        ${DATASET_RUNTIME_DIR}
        ${DATASET_PREPROCESS_DIR})
endforeach()

target_include_directories(triangulate_single_polygon PRIVATE ${SRC_DIR} ${DATASET_COMMON_DIR})

target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)
target_link_libraries(raytracer_filter_refine PRIVATE ${CUDA_LIBRARIES} CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system)
target_link_libraries(preprocess_dataset PRIVATE CGAL::CGAL tinyobjloader::tinyobjloader Boost::filesystem Boost::system ${CUDA_LIBRARIES})
target_link_libraries(triangulate_single_polygon PRIVATE CGAL::CGAL Boost::filesystem Boost::system)

target_compile_definitions(raytracer PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_filter_refine PRIVATE INCLUDE_OPTIX)

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_filter_refine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(triangulate_single_polygon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(preprocess_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

# -----------------------------------------------------------------------
# 6. Post-Build: Copy PTX & Fix DLLs (Robust Version)
# -----------------------------------------------------------------------

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)
add_dependencies(raytracer_filter_refine copy_ptx)

foreach(TARGET_NAME raytracer raytracer_filter_refine)
    # 1. Copy PTX
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PTX_FILE}"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/raytracing${OPTIX_MODULE_EXTENSION}"
        COMMENT "Copying PTX file..."
    )

    # 2. Copy DLLs from Conda Binary folder (With existence checks to prevent build failure)
    if(WIN32 AND DEFINED ENV{CONDA_PREFIX})
        set(CONDA_BIN_DIR "$ENV{CONDA_PREFIX}/Library/bin")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            # Check if file exists before copying to avoid error code 1
            COMMAND powershell.exe -Command "if (Test-Path '${CONDA_BIN_DIR}/boost_filesystem*.dll') { Copy-Item '${CONDA_BIN_DIR}/boost_filesystem*.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>' -Force }"
            COMMAND powershell.exe -Command "if (Test-Path '${CONDA_BIN_DIR}/boost_system*.dll') { Copy-Item '${CONDA_BIN_DIR}/boost_system*.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>' -Force }"
            COMMAND powershell.exe -Command "if (Test-Path '${CONDA_BIN_DIR}/nvoptix.dll') { Copy-Item '${CONDA_BIN_DIR}/nvoptix.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>' -Force }"
            COMMAND powershell.exe -Command "if (Test-Path '${CONDA_BIN_DIR}/cudart64*.dll') { Copy-Item '${CONDA_BIN_DIR}/cudart64*.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>' -Force }"
            COMMAND powershell.exe -Command "if (Test-Path '${CONDA_BIN_DIR}/zlib.dll') { Copy-Item '${CONDA_BIN_DIR}/zlib.dll' -Destination '$<TARGET_FILE_DIR:${TARGET_NAME}>' -Force }"
            COMMENT "Copying Boost/OptiX/CUDA/ZLib DLLs from ${CONDA_BIN_DIR} (if present)"
        )
    endif()
endforeach()