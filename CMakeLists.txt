cmake_minimum_required(VERSION 3.16)
project(edgevecdb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3 -fno-omit-frame-pointer")

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_CPU_BLAS "Enable CPU BLAS backend" ON)
option(USE_GPU_KOMP "Enable GPU Kompute backend" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build test programs" OFF)

# Kompute
set(KOMPUTE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/kompute")
if (NOT EXISTS ${KOMPUTE_PATH})
    message(WARNING "Kompute directory not found. Please clone it manually.")
endif()
include_directories(${KOMPUTE_PATH}/src/include)
include_directories(${CMAKE_BINARY_DIR}/external/kompute/src/shaders/glsl)
add_subdirectory(${KOMPUTE_PATH} ${CMAKE_BINARY_DIR}/external/kompute)

# Vulkan
set(VULKAN_SDK_VERSION 1.3.296.0)
set(VULKAN_INCLUDE_DIR "/usr/include/vulkan")
if(NOT EXISTS ${VULKAN_INCLUDE_DIR})
    message(FATAL_ERROR "Vulkan include directory not found: ${VULKAN_INCLUDE_DIR}. Please install Vulkan SDK.")
endif()

# OpenBLAS
find_library(OPENBLAS_LIB openblas
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)
if(NOT OPENBLAS_LIB)
    message(WARNING "OpenBLAS library not found. Please install libopenblas-dev.")
endif()

# 源文件收集
set(edgevecdb_SOURCES
    src/index/FlatIndex.cpp
)
set(edgevecdb_HEADERS
    src/index/Device.hpp
    src/index/FlatIndex.hpp
    src/index/MetricType.hpp
)
set(CPU_BLAS_SOURCES
    src/backend/cpu-blas/distance.cpp
    src/backend/cpu-blas/L2Norm.cpp
)
set(CPU_BLAS_HEADERS
    src/backend/cpu-blas/distance.hpp
    src/backend/cpu-blas/L2Norm.hpp
)
set(GPU_KOMPUTE_SOURCES
    src/backend/gpu-kompute/distance.cpp
    src/backend/gpu-kompute/readShader.cpp
    src/backend/gpu-kompute/L2Norm.cpp
)
set(GPU_KOMPUTE_HEADERS
    src/backend/gpu-kompute/distance.hpp
    src/backend/gpu-kompute/readShader.hpp
    src/backend/gpu-kompute/L2Norm.hpp
    src/backend/gpu-kompute/shader.hpp
)

if(USE_CPU_BLAS)
    list(APPEND edgevecdb_SOURCES ${CPU_BLAS_SOURCES})
    list(APPEND edgevecdb_HEADERS ${CPU_BLAS_HEADERS})
endif()
if(USE_GPU_KOMP)
    list(APPEND edgevecdb_SOURCES ${GPU_KOMPUTE_SOURCES})
    list(APPEND edgevecdb_HEADERS ${GPU_KOMPUTE_HEADERS})
endif()

add_library(edgevecdb STATIC ${edgevecdb_SOURCES} ${edgevecdb_HEADERS})

target_include_directories(edgevecdb PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/index
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu-blas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu-kompute
    ${KOMPUTE_PATH}/src/include
    ${VULKAN_INCLUDE_DIR}
)

if(USE_CPU_BLAS AND OPENBLAS_LIB)
    target_link_libraries(edgevecdb ${OPENBLAS_LIB})
endif()
if(USE_GPU_KOMP)
    target_link_libraries(edgevecdb kompute::kompute)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(edgevecdb OpenMP::OpenMP_CXX)

if(BUILD_TESTS)
    add_executable(testFlatIndex src/test/testFlatIndex.cpp)
    target_link_libraries(testFlatIndex edgevecdb)
    if(USE_GPU_KOMP)
        add_executable(testFlatIndexGpu src/test/testFlatIndexGpu.cpp)
        target_link_libraries(testFlatIndexGpu edgevecdb)
    endif()
endif()

if(BUILD_PYTHON_BINDINGS)
    # 标准查找 Python 和 pybind11
    find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
    find_package(pybind11 REQUIRED)
    
    message(STATUS "Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

    set(PYTHON_BINDING_SOURCES
        src/python/module.cpp
        src/python/numpy_helper.cpp
        src/python/common/enum_binding.cpp
        src/python/wrapper/pyFlatIndex.cpp
        src/python/binding/flatIndexBinding.cpp
        src/python/binding/L2RenormBinding.cpp
    )

    set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/edgevecdb)
    
    pybind11_add_module(edgevecdb_core MODULE ${PYTHON_BINDING_SOURCES})
    target_link_libraries(edgevecdb_core PRIVATE edgevecdb kompute::kompute ${OPENBLAS_LIB})
    target_include_directories(edgevecdb_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/index
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu-blas
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu-kompute
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    set_target_properties(edgevecdb_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
    )
    add_custom_command(
        TARGET edgevecdb_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/src/python/__init__.py
            ${PYTHON_MODULE_OUTPUT_DIR}/__init__.py
    )
endif()