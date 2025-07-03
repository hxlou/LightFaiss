
pybind11_add_module(kp src/main.cpp)

# 指定输出目录为当前源目录
set(PYTHON_MODULE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/../lightfaiss)

set_target_properties(
        kp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
)

include_directories(
        ${PROJECT_SOURCE_DIR}/include/)

target_link_libraries(
        kp PRIVATE
        kompute::kompute)
