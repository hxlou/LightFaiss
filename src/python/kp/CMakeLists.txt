
# pybind11_add_module(kp src/main.cpp)
add_library(kp MODULE src/main.cpp)
target_link_libraries(kp PRIVATE pybind11::module)
target_compile_definitions(kp PRIVATE MODULE_NAME=kp)
set_target_properties(kp PROPERTIES 
	PREFIX ""
	SUFFIX ".so"
)

# 指定输出目录为主项目的 edgevecdb 目录
set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/edgevecdb)

set_target_properties(
        kp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
)

include_directories(
        ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(
        kp PRIVATE
        kompute::kompute
        ${Python_LIBRARIES}
        ${Python_NumPy_LIBRARIES})
