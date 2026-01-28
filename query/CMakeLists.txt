cmake_minimum_required(VERSION 3.18)

# Force system compiler on Linux to avoid conda gcc/glibc conflicts
if(UNIX AND NOT APPLE AND NOT WIN32)
    if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED CMAKE_CXX_COMPILER)
        find_program(SYSTEM_CXX_COMPILER NAMES g++ PATHS /usr/bin /usr/local/bin NO_DEFAULT_PATH)
        if(SYSTEM_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${SYSTEM_CXX_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
            message(STATUS "Forcing system C++ compiler: ${SYSTEM_CXX_COMPILER}")
        endif()
    endif()
endif()

project(RaySpace3DQuery LANGUAGES CXX CUDA)

# Policy settings
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0167 NEW)

# -----------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "MSVC runtime" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIE")
    link_directories(/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------

# Detect conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: ${CONDA_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
    
    if(WIN32)
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/Library/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/Library/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/Library/lib")
    else()
        set(CONDA_CUDA_BIN "${CONDA_PREFIX}/bin")
        set(CONDA_CUDA_INCLUDE "${CONDA_PREFIX}/include")
        set(CONDA_CUDA_LIB "${CONDA_PREFIX}/lib")
        include_directories(SYSTEM "${CONDA_PREFIX}/include")
        link_directories("${CONDA_PREFIX}/lib")
    endif()
    
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

# -----------------------------------------------------------------------
# Add Common Library
# -----------------------------------------------------------------------
add_subdirectory(../common ${CMAKE_CURRENT_BINARY_DIR}/common)

# -----------------------------------------------------------------------
# Source Definitions
# -----------------------------------------------------------------------

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# CUDA Sources
set(CUDA_DIR ${SRC_DIR}/cuda)
set(CUDA_SOURCES ${CUDA_DIR}/raytracing.cu ${CUDA_DIR}/mesh_overlap.cu ${CUDA_DIR}/mesh_intersection.cu)
set(RESULT_COMPACTION_SOURCE ${CUDA_DIR}/result_compaction.cu)
set(MESH_OVERLAP_DEDUP_SOURCE ${CUDA_DIR}/mesh_overlap_deduplication.cu)
set(SCAN_UTILS_SOURCE ${CUDA_DIR}/scan_utils.cu)
set(ESTIMATED_INTERSECTION_SOURCE ${CUDA_DIR}/estimated_intersection.cu)
set(ESTIMATED_OVERLAP_SOURCE ${CUDA_DIR}/estimated_overlap.cu)

# Application Sources
set(RAYSPACE_MAIN_SOURCE ${SRC_DIR}/applications/rayspace_main.cpp)
set(FILTER_REFINE_SOURCE ${SRC_DIR}/applications/rayspace_filter_refine_main.cpp)
set(MESH_OVERLAP_APP_SOURCE ${SRC_DIR}/applications/rayspace_mesh_overlap_main.cpp)
set(MESH_INTERSECTION_APP_SOURCE ${SRC_DIR}/applications/rayspace_mesh_intersection_main.cpp)
set(ESTIMATED_INTERSECTION_APP_SOURCE ${SRC_DIR}/applications/raytracer_intersection_estimated_main.cpp)
set(ESTIMATED_OVERLAP_APP_SOURCE ${SRC_DIR}/applications/raytracer_overlap_estimated_main.cpp)

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
    ${RAYTRACING_DIR}/FilterRefine.cpp
    ${RAYTRACING_DIR}/MeshOverlapLauncher.cpp
    ${RAYTRACING_DIR}/MeshIntersectionLauncher.cpp)

# Runtime sources
set(RUNTIME_SOURCES
    ${SRC_DIR}/runtime/GeometryIO.cpp
    ${SRC_DIR}/runtime/PointIO.cpp)

# -----------------------------------------------------------------------
# PTX Compilation
# -----------------------------------------------------------------------

set(TIMER_SOURCE ${SRC_DIR}/timer.cpp)

set(PTX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

NVCUDA_COMPILE_MODULE(
    SOURCES       ${CUDA_SOURCES}
    TARGET_PATH   ${PTX_OUTPUT_DIR}
    EXTENSION     "${OPTIX_MODULE_EXTENSION}"
    GENERATED_FILES PTX_FILES
    NVCC_OPTIONS  "${OPTIX_PROGRAM_TARGET}" "--gpu-architecture=sm_75" "--relocatable-device-code=true" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}" "-I${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
)

# -----------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------

add_executable(raytracer 
    ${RAYSPACE_MAIN_SOURCE} 
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES} 
    ${GEOMETRY_SOURCES} 
    ${RAYTRACING_SOURCES} 
    ${RESULT_COMPACTION_SOURCE})

add_executable(raytracer_filter_refine 
    ${FILTER_REFINE_SOURCE} 
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES} 
    ${GEOMETRY_SOURCES} 
    ${RAYTRACING_SOURCES} 
    ${RESULT_COMPACTION_SOURCE})

add_executable(raytracer_mesh_overlap 
    ${MESH_OVERLAP_APP_SOURCE} 
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES} 
    ${GEOMETRY_SOURCES} 
    ${RAYTRACING_SOURCES} 
    ${RESULT_COMPACTION_SOURCE} 
    ${MESH_OVERLAP_DEDUP_SOURCE} 
    ${SCAN_UTILS_SOURCE})

add_executable(raytracer_mesh_intersection 
    ${MESH_INTERSECTION_APP_SOURCE} 
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES} 
    ${GEOMETRY_SOURCES} 
    ${RAYTRACING_SOURCES} 
    ${RESULT_COMPACTION_SOURCE} 
    ${MESH_OVERLAP_DEDUP_SOURCE} 
    ${SCAN_UTILS_SOURCE})

add_executable(raytracer_intersection_estimated
    ${ESTIMATED_INTERSECTION_APP_SOURCE}
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES}
    ${GEOMETRY_SOURCES}
    ${RAYTRACING_SOURCES}
    ${RESULT_COMPACTION_SOURCE}
    ${MESH_OVERLAP_DEDUP_SOURCE}
    ${SCAN_UTILS_SOURCE}
    ${ESTIMATED_INTERSECTION_SOURCE})

add_executable(raytracer_overlap_estimated
    ${ESTIMATED_OVERLAP_APP_SOURCE}
    ${RUNTIME_SOURCES}
    ${TIMER_SOURCE}
    ${OPTIX_SOURCES}
    ${GEOMETRY_SOURCES}
    ${RAYTRACING_SOURCES}
    ${RESULT_COMPACTION_SOURCE}
    ${MESH_OVERLAP_DEDUP_SOURCE}
    ${SCAN_UTILS_SOURCE}
    ${ESTIMATED_OVERLAP_SOURCE})

foreach(TARGET_NAME raytracer raytracer_filter_refine raytracer_mesh_overlap raytracer_mesh_intersection raytracer_intersection_estimated raytracer_overlap_estimated)
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CUDA_INCLUDE_DIRS} 
        ${OptiX_INCLUDE} 
        ${SRC_DIR}
        ${CUDA_DIR}
        ${OPTIX_DIR}
        ${GEOMETRY_DIR}
        ${RAYTRACING_DIR}
        ${SRC_DIR}/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
endforeach()

target_link_libraries(raytracer PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_link_libraries(raytracer_filter_refine PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_link_libraries(raytracer_mesh_overlap PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_link_libraries(raytracer_mesh_intersection PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_link_libraries(raytracer_intersection_estimated PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_link_libraries(raytracer_overlap_estimated PRIVATE 
    rayspace3d_common
    ${CUDA_LIBRARIES} 
    CGAL::CGAL 
    Boost::filesystem 
    Boost::system)

target_compile_definitions(raytracer PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_filter_refine PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_mesh_overlap PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_mesh_intersection PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_intersection_estimated PRIVATE INCLUDE_OPTIX)
target_compile_definitions(raytracer_overlap_estimated PRIVATE INCLUDE_OPTIX)

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_filter_refine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_mesh_overlap PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_mesh_intersection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_intersection_estimated PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(raytracer_overlap_estimated PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------
# Post-Build: Copy PTX
# -----------------------------------------------------------------------

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
set(MESH_OVERLAP_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_overlap${OPTIX_MODULE_EXTENSION}")
set(MESH_INTERSECTION_PTX_FILE "${PTX_OUTPUT_DIR}/mesh_intersection${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)
add_dependencies(raytracer_filter_refine copy_ptx)
add_dependencies(raytracer_mesh_overlap copy_ptx)
add_dependencies(raytracer_mesh_intersection copy_ptx)

foreach(TARGET_NAME raytracer raytracer_filter_refine)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PTX_FILE}"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>/raytracing${OPTIX_MODULE_EXTENSION}"
        COMMENT "Copying PTX file...")
endforeach()

add_custom_command(TARGET raytracer_mesh_overlap POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_mesh_overlap>/raytracing${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying raytracing.ptx...")

add_custom_command(TARGET raytracer_mesh_overlap POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MESH_OVERLAP_PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_mesh_overlap>/mesh_overlap${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying mesh_overlap.ptx...")

add_custom_command(TARGET raytracer_mesh_intersection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_mesh_intersection>/raytracing${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying raytracing.ptx...")

add_custom_command(TARGET raytracer_mesh_intersection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MESH_INTERSECTION_PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_mesh_intersection>/mesh_intersection${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying mesh_intersection.ptx...")

# -----------------------------------------------------------------------
# ESTIMATION Configuration
# -----------------------------------------------------------------------

# Intersection Estimated
add_dependencies(raytracer_intersection_estimated copy_ptx)

add_custom_command(TARGET raytracer_intersection_estimated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_intersection_estimated>/raytracing${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying raytracing.ptx...")

add_custom_command(TARGET raytracer_intersection_estimated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MESH_INTERSECTION_PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_intersection_estimated>/mesh_intersection${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying mesh_intersection.ptx...")

# Overlap Estimated
add_dependencies(raytracer_overlap_estimated copy_ptx)

add_custom_command(TARGET raytracer_overlap_estimated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_overlap_estimated>/raytracing${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying raytracing.ptx...")

add_custom_command(TARGET raytracer_overlap_estimated POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MESH_OVERLAP_PTX_FILE}"
    "$<TARGET_FILE_DIR:raytracer_overlap_estimated>/mesh_overlap${OPTIX_MODULE_EXTENSION}"
    COMMENT "Copying mesh_overlap.ptx...")
