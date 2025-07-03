add_executable(testFlatIndex testFlatIndex.cpp)
target_link_libraries(testFlatIndex PRIVATE index cpu-blas gpu-kompute)

add_executable(testFlatIndexGpu testFlatIndexGpu.cpp)
target_link_libraries(testFlatIndexGpu PRIVATE index cpu-blas gpu-kompute)

add_executable(testFlatIndexSL testFlatIndexSL.cpp)
target_link_libraries(testFlatIndexSL PRIVATE index cpu-blas gpu-kompute)

enable_testing()
add_test(NAME FlatIndexTest COMMAND testFlatIndex)