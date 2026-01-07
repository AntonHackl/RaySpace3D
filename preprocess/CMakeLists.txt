cmake_minimum_required(VERSION 3.25)
project(RaySpace3DPreprocess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# TinyObjLoader
find_package(tinyobjloader CONFIG QUIET)
if(NOT tinyobjloader_FOUND)
    message(FATAL_ERROR "TinyObjLoader not found. Install via conda: conda install tinyobjloader")
endif()

# Add common library
add_subdirectory(../common ${CMAKE_CURRENT_BINARY_DIR}/common)

# Include Eigen3 path
if(DEFINED ENV{CONDA_PREFIX})
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include/eigen3")
endif()

# TDBase library sources
set(TDBASE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tdbase_lib)
file(GLOB TDBASE_SPATIAL_SRCS ${TDBASE_LIB_DIR}/spatial/*.cpp)
file(GLOB TDBASE_STORAGE_SRCS ${TDBASE_LIB_DIR}/storage/*.cpp)
file(GLOB TDBASE_GEOMETRY_SRCS ${TDBASE_LIB_DIR}/geometry/*.cpp)
file(GLOB TDBASE_INDEX_SRCS ${TDBASE_LIB_DIR}/index/*.cpp)

set(TDBASE_SOURCES 
    ${TDBASE_SPATIAL_SRCS} 
    ${TDBASE_STORAGE_SRCS} 
    ${TDBASE_GEOMETRY_SRCS} 
    ${TDBASE_INDEX_SRCS}
)

# Dataset loader sources
set(DATASET_SOURCES
    src/DatasetLoaderFactory.cpp
    src/PolygonDatasetLoader.cpp
    src/ObjMeshDatasetLoader.cpp
    src/DtMeshDatasetLoader.cpp
    ${TDBASE_SOURCES}
)

# Preprocessing executable
add_executable(preprocess_dataset
    src/preprocess_dataset.cpp
    ${DATASET_SOURCES}
)

target_include_directories(preprocess_dataset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TDBASE_LIB_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    $ENV{CONDA_PREFIX}/include
    $ENV{CONDA_PREFIX}/targets/x86_64-linux/include
)

# Add CUDA library path even though we don't link CUDA at runtime
if(DEFINED ENV{CONDA_PREFIX})
    target_link_directories(preprocess_dataset PRIVATE $ENV{CONDA_PREFIX}/lib)
endif()

target_link_libraries(preprocess_dataset PRIVATE
    rayspace3d_common
    CGAL::CGAL
    tinyobjloader::tinyobjloader
    Boost::filesystem
    Boost::system
    OpenMP::OpenMP_CXX
    cudart  # CUDA runtime for pinned memory allocation
)

target_compile_definitions(preprocess_dataset PRIVATE
    BOOST_HAS_THREADS
    CGAL_HAS_THREADS
    CGAL_DONT_USE_LAZY_KERNEL
)

set_target_properties(preprocess_dataset PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
