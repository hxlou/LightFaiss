# 假设你的源码是 distance.cpp 和 distance_utils.cpp
add_library(cpu-blas STATIC distance.cpp)

# 链接OpenBLAS和Kompute
target_link_libraries(cpu-blas PUBLIC openblas kompute::kompute)

# 包含头文件目录
target_include_directories(cpu-blas PUBLIC
    ${OPENBLAS_PATH}
    ${KOMPUTE_PATH}/src/include
)