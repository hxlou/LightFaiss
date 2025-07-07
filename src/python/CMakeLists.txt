# Python bindings CMakeLists.txt
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

# 收集所有源文件
set(PYTHON_BINDING_SOURCES
    module.cpp
    numpy_helper.cpp

    # 通用绑定
    common/enum_binding.cpp

    # FlatIndex相关
    wrapper/pyFlatIndex.cpp
    binding/flatIndexBinding.cpp
    binding/L2RenormBinding.cpp
)

# 指定输出目录为当前源目录
set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/lightfaiss)

# 创建Python模块
pybind11_add_module(lightfaiss_py SHARED ${PYTHON_BINDING_SOURCES})

# 设置输出目录
set_target_properties(lightfaiss_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
)

# 设置编译属性
target_compile_definitions(lightfaiss_py PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 设置包含目录
target_include_directories(lightfaiss_py PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../index
    ${CMAKE_CURRENT_SOURCE_DIR}/../backend
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/binding
    ${Python_NumPy_INCLUDE_DIRS}
)

# 链接必要的库
target_link_libraries(lightfaiss_py PRIVATE 
        index 
        backend
        kompute::kompute
        ${Python_LIBRARIES}  
        ${Python_NumPy_LIBRARIES})

add_custom_command(
    TARGET lightfaiss_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${PYTHON_MODULE_OUTPUT_DIR}/__init__.py
)

# 添加kp子目录
add_subdirectory(kp)