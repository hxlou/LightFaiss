#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

class NumpyHelper {
public:
    // 将numpy数组转换为float指针和大小信息
    static std::pair<const float*, size_t> numpy_to_float_ptr(py::array_t<float> input);
    
    // 将numpy数组转换为uint64_t指针和大小信息
    static std::pair<const uint64_t*, size_t> numpy_to_uint64_ptr(py::array_t<uint64_t> input);
    
    // 创建numpy数组从float指针
    static py::array_t<float> float_ptr_to_numpy(const float* data, size_t size);
    
    // 创建numpy数组从uint64_t指针
    static py::array_t<uint64_t> uint64_ptr_to_numpy(const uint64_t* data, size_t size);
    
    // 创建2D numpy数组
    static py::array_t<float> create_2d_float_array(size_t rows, size_t cols);
    static py::array_t<uint64_t> create_2d_uint64_array(size_t rows, size_t cols);
    
    // 验证numpy数组的形状和类型
    static void validate_float_array(py::array_t<float> arr, const std::string& name);
    static void validate_uint64_array(py::array_t<uint64_t> arr, const std::string& name);
};