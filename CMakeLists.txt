cmake_minimum_required(VERSION 3.16)
project(MyVectorDB LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 选项：启用/禁用不同后端
option(USE_CPU_BLAS "Enable CPU BLAS backend" ON)
option(USE_GPU_KOMP "Enable GPU Kompute backend" ON)
option(USE_NPU_HEXAGON "Enable NPU Hexagon backend" OFF)

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

add_subdirectory(${OPENBLAS_PATH} ${CMAKE_BINARY_DIR}/external/OpenBLAS)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src)