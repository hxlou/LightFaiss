add_subdirectory(cpu-blas)
add_subdirectory(gpu-kompute)
# NPU Hexagon 后端未实现，暂不添加

add_library(backend INTERFACE)
target_link_libraries(backend INTERFACE cpu-blas gpu-kompute)