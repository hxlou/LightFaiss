#include "src/index/FlatIndex.hpp"
#include "src/python/wrapper/pyFlatIndex.hpp"
#include "src/python/numpy_helper.hpp"

#include <stdexcept>

PyFlatIndex::PyFlatIndex(uint64_t dim, uint64_t capacity, bool isFloat16, MetricType metricType, kp::Manager* mgr) 
    : index_(std::make_unique<FlatIndex>(dim, capacity, isFloat16, metricType, mgr)) {}

PyFlatIndex::PyFlatIndex(uint64_t dim) 
    : index_(std::make_unique<FlatIndex>(dim)) {}

void PyFlatIndex::add_vectors(py::array_t<float> vectors) {
    auto buf = vectors.request();
    
    // 简单直接的验证
    if (buf.format != py::format_descriptor<float>::format()) {
        throw std::runtime_error("Expected float32 array");
    }
    
    uint64_t n = buf.shape[0];
    uint64_t dim = buf.shape[1];
    
    if (dim != index_->getDim()) {
        throw std::runtime_error("Dimension mismatch: expected " + 
                                std::to_string(index_->getDim()) + ", got " + std::to_string(dim));
    }
    
    const float* data = static_cast<const float*>(buf.ptr);
    index_->addVector(data, n);
}

uint64_t PyFlatIndex::get_num() const { 
    return index_->getNum(); 
}

uint64_t PyFlatIndex::get_dim() const { 
    return index_->getDim(); 
}

uint64_t PyFlatIndex::get_capacity() const { 
    return index_->getCapacity(); 
}

bool PyFlatIndex::is_float16() const { 
    return index_->isFloat16(); 
}

// py::tuple PyFlatIndex::query(py::array_t<float> queries, uint64_t k, DeviceType device) {
//     py::buffer_info buf = queries.request();
    
//     uint64_t n = buf.shape[0];
//     uint64_t dim = buf.shape[1];
    
//     if (dim != index_->getDim()) {
//         throw std::runtime_error("Query dimension mismatch: expected " + 
//                                 std::to_string(index_->getDim()) + ", got " + std::to_string(dim));
//     }
    
//     // 创建输出数组
//     auto results = NumpyHelper::create_2d_uint64_array(n, k);
//     auto distances = NumpyHelper::create_2d_float_array(n, k);
    
//     py::buffer_info results_buf = results.request();
//     py::buffer_info distances_buf = distances.request();
    
//     float* query_data = static_cast<float*>(buf.ptr);
//     uint64_t* results_data = static_cast<uint64_t*>(results_buf.ptr);
//     float* distances_data = static_cast<float*>(distances_buf.ptr);
    
//     index_->query(n, k, device, query_data, results_data, distances_data);
    
//     return py::make_tuple(results, distances);
// }

py::tuple PyFlatIndex::query_range(py::array_t<float> queries, uint64_t k, uint64_t start, uint64_t end, 
                                  DeviceType device) {
    py::buffer_info buf = queries.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Query vectors must be 2D array (n_queries, dim)");
    }
    
    uint64_t nQuery = buf.shape[0];
    uint64_t dim = buf.shape[1];
    
    if (dim != index_->getDim()) {
        throw std::runtime_error("Query dimension mismatch: expected " + 
                                std::to_string(index_->getDim()) + ", got " + std::to_string(dim));
    }
    
    // 创建输出数组
    auto results = NumpyHelper::create_2d_uint64_array(nQuery, k);
    auto distances = NumpyHelper::create_2d_float_array(nQuery, k);
    
    py::buffer_info results_buf = results.request();
    py::buffer_info distances_buf = distances.request();
    
    const float* query_data = static_cast<const float*>(buf.ptr);
    uint64_t* results_data = static_cast<uint64_t*>(results_buf.ptr);
    float* distances_data = static_cast<float*>(distances_buf.ptr);
    
    index_->query(k, start, end, device, nQuery, query_data, 
                 results_data, distances_data);
    
    return py::make_tuple(results, distances);
}

py::array_t<float> PyFlatIndex::reconstruct(uint64_t idx) {
    // 创建一个新的numpy数组来存储重建的向量
    auto result = py::array_t<float>(index_->getDim());
    py::buffer_info buf = result.request();
    float* data = static_cast<float*>(buf.ptr);
    
    index_->reconstruct(idx, data);
    return result;
}

int PyFlatIndex::save(const std::string& filename) {
    return index_->save(filename);
}

int PyFlatIndex::load(const std::string& filename) {
    return index_->load(filename);
}