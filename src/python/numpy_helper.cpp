#include "src/python/numpy_helper.hpp"
#include <stdexcept>

// std::pair<const float*, size_t> NumpyHelper::numpy_to_float_ptr(py::array_t<float> input) {
//     py::buffer_info buf_info = input.request();
    
//     if (buf_info.format != py::format_descriptor<float>::format()) {
//         throw std::runtime_error("Input array must be of type float32");
//     }
    
//     return std::make_pair(static_cast<const float*>(buf_info.ptr), buf_info.size);
// }

// std::pair<const uint64_t*, size_t> NumpyHelper::numpy_to_uint64_ptr(py::array_t<uint64_t> input) {
//     py::buffer_info buf_info = input.request();
    
//     if (buf_info.format != py::format_descriptor<uint64_t>::format()) {
//         throw std::runtime_error("Input array must be of type uint64");
//     }
    
//     return std::make_pair(static_cast<const uint64_t*>(buf_info.ptr), buf_info.size);
// }

// py::array_t<float> NumpyHelper::float_ptr_to_numpy(const float* data, size_t size) {
//     // 方法2：使用 py::handle 来处理 base object
//     return py::array_t<float>(
//         size,                                   // shape
//         {sizeof(float)},                       // strides
//         data,                                  // data pointer
//         py::handle()                           // 空的 handle 作为 base object
//     );
// }

// py::array_t<uint64_t> NumpyHelper::uint64_ptr_to_numpy(const uint64_t* data, size_t size) {
//     // 方法2：使用 py::handle 来处理 base object
//     return py::array_t<uint64_t>(
//         size,                                   // shape
//         {sizeof(uint64_t)},                    // strides
//         data,                                  // data pointer
//         py::handle()                           // 空的 handle 作为 base object
//     );
// }

py::array_t<float> NumpyHelper::create_2d_float_array(size_t rows, size_t cols) {
    return py::array_t<float>(
        {rows, cols},                          // shape
        {sizeof(float) * cols, sizeof(float)} // strides
    );
}

py::array_t<uint64_t> NumpyHelper::create_2d_uint64_array(size_t rows, size_t cols) {
    return py::array_t<uint64_t>(
        {rows, cols},                              // shape
        {sizeof(uint64_t) * cols, sizeof(uint64_t)} // strides
    );
}

// void NumpyHelper::validate_float_array(py::array_t<float> arr, const std::string& name) {
//     py::buffer_info buf = arr.request();
//     if (buf.format != py::format_descriptor<float>::format()) {
//         throw std::runtime_error(name + " must be a float32 array");
//     }
// }

// void NumpyHelper::validate_uint64_array(py::array_t<uint64_t> arr, const std::string& name) {
//     py::buffer_info buf = arr.request();
//     if (buf.format != py::format_descriptor<uint64_t>::format()) {
//         throw std::runtime_error(name + " must be a uint64 array");
//     }
// }