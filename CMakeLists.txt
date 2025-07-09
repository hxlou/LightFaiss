cmake_minimum_required(VERSION 3.16)
project(edgevecdb LANGUAGES CXX)

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
option(USE_GPU_KOMP "Enable GPU Kompute backend" OFF)
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
# if (NOT EXISTS ${OPENBLAS_PATH})
#     message(WARNING "OpenBLAS directory not found. Cloning from GitHub...")
#     execute_process(
#             COMMAND git clone https://github.com/zju-aces-aios/OpenBLAS.git -b android-port ${OPENBLAS_PATH}
#             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#     )
# endif ()
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

# execute_process(
#         COMMAND make
#         HOSTCC=gcc
#         CC=${NDK_TOOLCHAIN_PATH}/aarch64-linux-android24-clang
#         BUILD_WITHOUT_LAPACKE=0
#         C_LAPACK=1
#         NO_CBLAS=0
#         NO_LAPACK=0
#         USE_OPENMP=1
#         NOFORTRAN=1
#         AR=${NDK_TOOLCHAIN_PATH}/llvm-ar
#         TARGET=CORTEXA57
#         RANLIB=${NDK_TOOLCHAIN_PATH}/llvm-ranlib
#         WORKING_DIRECTORY ${OPENBLAS_PATH}
#         RESULT_VARIABLE result
#         OUTPUT_FILE ${OUTPUT_FILE}
#         ERROR_FILE ${ERROR_FILE}
# )

# if (NOT result EQUAL 0)
#     message(FATAL_ERROR "Make command failed with result: ${result}. Check logs at: ${OUTPUT_FILE} and ${ERROR_FILE}")
# endif ()




# 收集所有源文件
set(edgevecdb_SOURCES
    # 核心索引文件
    src/index/FlatIndex.cpp
)

# 收集所有头文件
set(edgevecdb_HEADERS
    # 核心头文件
    src/index/Device.hpp
    src/index/FlatIndex.hpp
    src/index/MetricType.hpp
)

# CPU BLAS 后端文件
set(CPU_BLAS_SOURCES
    src/backend/cpu-blas/distance.cpp
    src/backend/cpu-blas/L2Norm.cpp
)
set(CPU_BLAS_HEADERS
    src/backend/cpu-blas/distance.hpp
    src/backend/cpu-blas/L2Norm.hpp
)

# GPU Kompute 后端文件
set(GPU_KOMPUTE_SOURCES
    src/backend/gpu-kompute/distance.cpp
    src/backend/gpu-kompute/readShader.cpp
    src/backend/gpu-kompute/L2Norm.cpp
)
set(GPU_KOMPUTE_HEADERS
    src/backend/gpu-kompute/distance.hpp
    src/backend/gpu-kompute/readShader.hpp
    src/backend/gpu-kompute/L2Norm.hpp
)

# 根据选项添加后端文件
if(USE_CPU_BLAS)
    list(APPEND edgevecdb_SOURCES ${CPU_BLAS_SOURCES})
    list(APPEND edgevecdb_HEADERS ${CPU_BLAS_HEADERS})
endif()

if(USE_GPU_KOMP)
    list(APPEND edgevecdb_SOURCES ${GPU_KOMPUTE_SOURCES})
    list(APPEND edgevecdb_HEADERS ${GPU_KOMPUTE_HEADERS})
endif()

# 创建edgevecdb静态库
add_library(edgevecdb STATIC ${edgevecdb_SOURCES} ${edgevecdb_HEADERS})

# Android特定设置
if(ANDROID)
    target_compile_definitions(edgevecdb PRIVATE
        ANDROID=1
        __ANDROID__=1
    )
    # 确保使用C++17标准
    set_target_properties(edgevecdb PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

# 设置头文件包含目录
target_include_directories(edgevecdb PUBLIC 
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
    # if(EXISTS ${OPENBLAS_PATH}/libopenblas.a)
        # target_link_libraries(edgevecdb ${OPENBLAS_PATH}/libopenblas.a)
	target_link_libraries(edgevecdb /home/ubuntu/chaquopy/server/pypi/libopenblas.a)
    # else()
        # message(WARNING "OpenBLAS static library not found at ${OPENBLAS_PATH}/libopenblas.a")
    # endif()
endif()

if(USE_GPU_KOMP)
    # 链接Kompute（通过add_subdirectory已经可用）
    target_link_libraries(edgevecdb kompute::kompute)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(edgevecdb OpenMP::OpenMP_CXX)

# 可选：编译测试程序（如果需要的话）
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    add_executable(testFlatIndex src/test/testFlatIndex.cpp)
    target_link_libraries(testFlatIndex edgevecdb)
    
    if(USE_GPU_KOMP)
        add_executable(testFlatIndexGpu src/test/testFlatIndexGpu.cpp)
        target_link_libraries(testFlatIndexGpu edgevecdb)
    endif()
endif()



# 如果启用Python绑定，需要找到Python和pybind11
if(BUILD_PYTHON_BINDINGS)
	    if(ANDROID)
        message(STATUS "Android build detected, using manual Python configuration")
        
        # 调试：显示所有传入的Python相关变量
        message(STATUS "=== Python Variables Debug ===")
        message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
        message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
        message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
        message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
        message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
        message(STATUS "PYTHON_LIBRARY: ${PYTHON_LIBRARY}")
        message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
        message(STATUS "PYTHON_NUMPY_INCLUDE_DIR: ${PYTHON_NUMPY_INCLUDE_DIR}")
        message(STATUS "==============================")
        
        # 对于 Android，我们期望从 setup.py 传递所有必要的 Python 变量
        # 检查新格式和旧格式的变量名
        if(NOT DEFINED Python_EXECUTABLE AND NOT DEFINED PYTHON_EXECUTABLE)
            message(FATAL_ERROR "Python_EXECUTABLE or PYTHON_EXECUTABLE must be set for Android builds")
        endif()
        if(NOT DEFINED Python_INCLUDE_DIRS AND NOT DEFINED PYTHON_INCLUDE_DIR)
            message(FATAL_ERROR "Python_INCLUDE_DIRS or PYTHON_INCLUDE_DIR must be set for Android builds")
        endif()
        if(NOT DEFINED Python_NumPy_INCLUDE_DIRS AND NOT DEFINED PYTHON_NUMPY_INCLUDE_DIR)
            message(FATAL_ERROR "Python_NumPy_INCLUDE_DIRS or PYTHON_NUMPY_INCLUDE_DIR must be set for Android builds")
        endif()
        
        # 统一变量名 - 使用新格式
        if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
            set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
        endif()
        if(DEFINED PYTHON_INCLUDE_DIR AND NOT DEFINED Python_INCLUDE_DIRS)
            set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
        endif()
        if(DEFINED PYTHON_LIBRARY AND NOT DEFINED Python_LIBRARIES)
            set(Python_LIBRARIES ${PYTHON_LIBRARY})
        endif()
        if(DEFINED PYTHON_NUMPY_INCLUDE_DIR AND NOT DEFINED Python_NumPy_INCLUDE_DIRS)
            set(Python_NumPy_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR})
        endif()
        
        # 为了兼容性，设置一些 pybind11 期望的变量
        set(Python3_EXECUTABLE ${Python_EXECUTABLE})
        set(Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
        set(Python3_LIBRARIES ${Python_LIBRARIES})
        set(Python3_FOUND TRUE)
        set(Python3_Interpreter_FOUND TRUE)
        set(Python3_Development_FOUND TRUE)
        
        message(STATUS "Final Python executable: ${Python_EXECUTABLE}")
        message(STATUS "Final Python include dirs: ${Python_INCLUDE_DIRS}")
        message(STATUS "Final Python NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
        
    else()
        # 非 Android 构建，使用标准 Python 查找
        find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
        message(STATUS "Python executable: ${Python_EXECUTABLE}")
        message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
        message(STATUS "Python NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
		find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
        message(STATUS "Python executable: ${Python_EXECUTABLE}")
        message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
        message(STATUS "Python NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        find_package(pybind11 REQUIRED)
    endif()



	set(Python_ARTIFACTS_INTERACTIVE ON)
    # find_package(Python COMPONENTS Interpreter NumPy REQUIRED)
	# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
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




	# 收集所有源文件
	set(PYTHON_BINDING_SOURCES
		./src/python/module.cpp
		./src/python/numpy_helper.cpp

		# 通用绑定
		./src/python/common/enum_binding.cpp

		# FlatIndex相关
		./src/python/wrapper/pyFlatIndex.cpp
		./src/python/binding/flatIndexBinding.cpp
		./src/python/binding/L2RenormBinding.cpp
	)

	# 指定输出目录为当前源目录
	set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/edgevecdb)

	# 创建Python模块
	# pybind11_add_module(edgevecdb_core SHARED ${PYTHON_BINDING_SOURCES})
	# 对于 Android，使用更兼容的方式创建模块
	add_library(edgevecdb_core MODULE ${PYTHON_BINDING_SOURCES})
	target_link_libraries(edgevecdb_core PRIVATE pybind11::module)
	target_compile_definitions(edgevecdb_core PRIVATE MODULE_NAME=edgevecdb_core)
	set_target_properties(edgevecdb_core PROPERTIES 
		PREFIX ""
		SUFFIX ".so"
	)

	target_compile_definitions(edgevecdb_core PRIVATE FMT_USE_ANDROID_LOG)

	# 设置输出目录
	set_target_properties(edgevecdb_core PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
	)

	# 设置编译属性
	target_compile_definitions(edgevecdb_core PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

	# 设置包含目录
	target_include_directories(edgevecdb_core PRIVATE 
		${CMAKE_CURRENT_SOURCE_DIR}
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/src/index
		${CMAKE_CURRENT_SOURCE_DIR}/src/backend
		${CMAKE_CURRENT_SOURCE_DIR}/python
		${CMAKE_CURRENT_SOURCE_DIR}/python/common
		${CMAKE_CURRENT_SOURCE_DIR}/python/wrapper
		${CMAKE_CURRENT_SOURCE_DIR}/python/binding
		${Python_NumPy_INCLUDE_DIRS}

	)

	# 链接必要的库
	target_link_libraries(edgevecdb_core PRIVATE 
			edgevecdb
			kompute::kompute
			${Python_LIBRARIES}  
			${Python_NumPy_LIBRARIES}
			-lc)

	add_custom_command(
		TARGET edgevecdb_core POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
			${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
			${PYTHON_MODULE_OUTPUT_DIR}/__init__.py
	)

	# 添加kp子目录
	add_subdirectory(./src/python/kp)
endif()

# add_subdirectory(${OPENBLAS_PATH} ${CMAKE_BINARY_DIR}/external/OpenBLAS)

# include_directories(${CMAKE_SOURCE_DIR})

# add_subdirectory(src)