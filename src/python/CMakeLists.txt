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
    
    # 当你添加其他索引时，在这里添加相应的源文件
    # wrappers/py_hnsw_index.cpp
    # bindings/hnsw_index_binding.cpp
)

# 创建Python模块
pybind11_add_module(lightfaiss_py ${PYTHON_BINDING_SOURCES})

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

# 设置Python模块属性
# set_target_properties(lightfaiss_py PROPERTIES
#     CXX_VISIBILITY_PRESET "hidden"
#     VISIBILITY_INLINES_HIDDEN YES
# )

# 链接必要的库
target_link_libraries(lightfaiss_py PRIVATE 
        -Wl,--whole-archive index -Wl,--no-whole-archive
        index 
        cpu-blas 
        gpu-kompute  
        ${Python_LIBRARIES}  
        ${Python_NumPy_LIBRARIES})
