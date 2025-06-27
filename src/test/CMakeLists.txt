add_executable(testFlatIndex testFlatIndex.cpp)
target_link_libraries(testFlatIndex PRIVATE index cpu-blas gpu-kompute)
enable_testing()
add_test(NAME FlatIndexTest COMMAND testFlatIndex)