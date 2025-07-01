cmake_minimum_required(VERSION 3.16)
project(LightFaiss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 调试信息
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

# 全局开启位置无关代码（PIC）
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 选项：启用/禁用不同后端
option(USE_CPU_BLAS "Enable CPU BLAS backend" ON)
option(USE_GPU_KOMP "Enable GPU Kompute backend" ON)
option(USE_NPU_HEXAGON "Enable NPU Hexagon backend" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

set(BUILD_SHARED_LIBS ON)

# Kompute路径（你可以将 external 目录放在 myDB 根目录下）
set(KOMPUTE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/kompute")
if (NOT EXISTS ${KOMPUTE_PATH})
    message(WARNING "Kompute directory not found. Cloning from GitHub...")
    execute_process(
            COMMAND git clone https://github.com/zju-aces-aios/kompute.git -b android-port ${KOMPUTE_PATH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif ()

set(VULKAN_SDK_VERSION 1.3.296.0)
set(VULKAN_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan)
# For Windows and macOS, set this after installing Vulkan SDK
# For example, set(VULKAN_INCLUDE_DIR /Users/example/VulkanSDK/1.3.296.0/macOS/include)
set(VULKAN_INCLUDE_DIR /Users/zlaa/VulkanSDK/1.3.296.0/macOS/include)
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
    if(NOT EXISTS ${VULKAN_INCLUDE_DIR})
        file(DOWNLOAD
                https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz
                ${CMAKE_BINARY_DIR}/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz
                SHOW_PROGRESS
        )

        file(MAKE_DIRECTORY ${VULKAN_SDK_DIR})

        execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xJf ${CMAKE_BINARY_DIR}/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz
                WORKING_DIRECTORY ${VULKAN_SDK_DIR}
        )
    endif()
    set(VULKAN_INCLUDE_DIR ${VULKAN_SDK_DIR}/${VULKAN_SDK_VERSION}/x86_64/include)
endif ()
if(NOT EXISTS ${VULKAN_INCLUDE_DIR})
    message(FATAL_ERROR "Vulkan include directory not found: ${VULKAN_INCLUDE_DIR}. Please download and install Vulkan SDK version ${VULKAN_SDK_VERSION} from https://vulkan.lunarg.com/sdk/home")
endif()
option(KOMPUTE_OPT_GIT_TAG "The tag of the repo to use for the example" v0.9.0)
option(KOMPUTE_OPT_FROM_SOURCE "Whether to build example from source or from git fetch repo" ON)

set(KOMPUTE_OPT_ANDROID_BUILD ON)
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON)

include_directories(${KOMPUTE_PATH}/src/include)
include_directories(${CMAKE_BINARY_DIR}/external/kompute/src/shaders/glsl)

add_subdirectory(${KOMPUTE_PATH} ${CMAKE_BINARY_DIR}/external/kompute)

# 查找BLAS库（OpenBLAS优先，其次系统BLAS）
set(OPENBLAS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/OpenBLAS)
if (NOT EXISTS ${OPENBLAS_PATH})
    message(WARNING "OpenBLAS directory not found. Cloning from GitHub...")
    execute_process(
            COMMAND git clone https://github.com/zju-aces-aios/OpenBLAS.git -b android-port ${OPENBLAS_PATH}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif ()
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Darwin")
    set(NDK_HOST "darwin-x86_64")
elseif(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Linux")
    set(NDK_HOST "linux-x86_64")
elseif(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(NDK_HOST "windows-x86_64")
else()
    message(FATAL_ERROR "Unsupported host system: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

set(NDK_TOOLCHAIN_PATH "${ANDROID_NDK}/toolchains/llvm/prebuilt/${NDK_HOST}/bin")
set(OUTPUT_FILE ${CMAKE_BINARY_DIR}/OpenBLAS_build_output.log)
set(ERROR_FILE ${CMAKE_BINARY_DIR}/OpenBLAS_build_error.log)

execute_process(
        COMMAND make
        HOSTCC=gcc
        CC=${NDK_TOOLCHAIN_PATH}/aarch64-linux-android35-clang
        BUILD_WITHOUT_LAPACKE=0
        C_LAPACK=1
        NO_CBLAS=0
        NO_LAPACK=0
        USE_OPENMP=1
        NOFORTRAN=1
        AR=${NDK_TOOLCHAIN_PATH}/llvm-ar
        TARGET=CORTEXA57
        RANLIB=${NDK_TOOLCHAIN_PATH}/llvm-ranlib
        WORKING_DIRECTORY ${OPENBLAS_PATH}
        RESULT_VARIABLE result
        OUTPUT_FILE ${OUTPUT_FILE}
        ERROR_FILE ${ERROR_FILE}
)

if (NOT result EQUAL 0)
    message(FATAL_ERROR "Make command failed with result: ${result}. Check logs at: ${OUTPUT_FILE} and ${ERROR_FILE}")
endif ()




# 收集所有源文件
set(LIGHTFAISS_SOURCES
    # 核心索引文件
    src/index/FlatIndex.cpp
)

# 收集所有头文件
set(LIGHTFAISS_HEADERS
    # 核心头文件
    src/index/Device.hpp
    src/index/FlatIndex.hpp
    src/index/MetricType.hpp
)

# CPU BLAS 后端文件
set(CPU_BLAS_SOURCES
    src/backend/cpu-blas/distance.cpp
)
set(CPU_BLAS_HEADERS
    src/backend/cpu-blas/distance.hpp
)

# GPU Kompute 后端文件
set(GPU_KOMPUTE_SOURCES
    src/backend/gpu-kompute/distance.cpp
    src/backend/gpu-kompute/readShader.cpp
)
set(GPU_KOMPUTE_HEADERS
    src/backend/gpu-kompute/distance.hpp
    src/backend/gpu-kompute/readShader.hpp
)

# 根据选项添加后端文件
if(USE_CPU_BLAS)
    list(APPEND LIGHTFAISS_SOURCES ${CPU_BLAS_SOURCES})
    list(APPEND LIGHTFAISS_HEADERS ${CPU_BLAS_HEADERS})
endif()

if(USE_GPU_KOMP)
    list(APPEND LIGHTFAISS_SOURCES ${GPU_KOMPUTE_SOURCES})
    list(APPEND LIGHTFAISS_HEADERS ${GPU_KOMPUTE_HEADERS})
endif()

# 创建lightfaiss静态库
add_library(lightfaiss STATIC ${LIGHTFAISS_SOURCES} ${LIGHTFAISS_HEADERS})

# Android特定设置
if(ANDROID)
    target_compile_definitions(lightfaiss PRIVATE
        ANDROID=1
        __ANDROID__=1
    )
    # 确保使用C++17标准
    set_target_properties(lightfaiss PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

# 设置头文件包含目录
target_include_directories(lightfaiss PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/index
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu-blas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu-kompute
    ${KOMPUTE_PATH}/src/include
    ${VULKAN_INCLUDE_DIR}
)

# 链接库
if(USE_CPU_BLAS)
    # 链接OpenBLAS
    if(EXISTS ${OPENBLAS_PATH}/libopenblas.a)
        target_link_libraries(lightfaiss ${OPENBLAS_PATH}/libopenblas.a)
    else()
        message(WARNING "OpenBLAS static library not found at ${OPENBLAS_PATH}/libopenblas.a")
    endif()
endif()

if(USE_GPU_KOMP)
    # 链接Kompute（通过add_subdirectory已经可用）
    target_link_libraries(lightfaiss kompute::kompute)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(lightfaiss OpenMP::OpenMP_CXX)

# 可选：编译测试程序（如果需要的话）
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    add_executable(testFlatIndex src/test/testFlatIndex.cpp)
    target_link_libraries(testFlatIndex lightfaiss)
    
    if(USE_GPU_KOMP)
        add_executable(testFlatIndexGpu src/test/testFlatIndexGpu.cpp)
        target_link_libraries(testFlatIndexGpu lightfaiss)
    endif()
endif()



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

# add_subdirectory(${OPENBLAS_PATH} ${CMAKE_BINARY_DIR}/external/OpenBLAS)

# include_directories(${CMAKE_SOURCE_DIR})

# add_subdirectory(src)