cmake_minimum_required(VERSION 3.16)
project(MyVectorDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 全局开启位置无关代码（PIC）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 选项：启用/禁用不同后端
option(USE_CPU_BLAS "Enable CPU BLAS backend" ON)
option(USE_GPU_KOMP "Enable GPU Kompute backend" ON)
option(USE_NPU_HEXAGON "Enable NPU Hexagon backend" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

set(BUILD_SHARED_LIBS ON)

# Kompute路径（你可以将 external 目录放在 myDB 根目录下）
set(KOMPUTE_PATH "${CMAKE_SOURCE_DIR}/external/kompute")
if (NOT EXISTS ${KOMPUTE_PATH})
    message(WARNING "Kompute directory not found. Cloning from GitHub...")
    execute_process(
            COMMAND git clone https://github.com/zju-aces-aios/kompute.git -b android-port ${KOMPUTE_PATH}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif ()

add_subdirectory(${KOMPUTE_PATH} ${CMAKE_BINARY_DIR}/external/kompute)

# 查找BLAS库（OpenBLAS优先，其次系统BLAS）
set(OPENBLAS_PATH ${CMAKE_SOURCE_DIR}/external/OpenBLAS)
if (NOT EXISTS ${OPENBLAS_PATH})
    message(WARNING "OpenBLAS directory not found. Cloning from GitHub...")
    execute_process(
            COMMAND git clone https://github.com/zju-aces-aios/OpenBLAS.git -b android-port ${OPENBLAS_PATH}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif ()

# 如果启用Python绑定，需要找到Python和pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
    find_package(pybind11 REQUIRED HINTS ${Python_SITELIB}/pybind11/share/cmake/pybind11)
    
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, trying to find via pip...")
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE pybind11_RESULT
        )
        if(pybind11_RESULT EQUAL 0)
            find_package(pybind11 REQUIRED HINTS ${pybind11_DIR})
        else()
            message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
        endif()
    endif()
    
    message(STATUS "Python bindings will be built")
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
endif()

add_subdirectory(${OPENBLAS_PATH} ${CMAKE_BINARY_DIR}/external/OpenBLAS)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(src)