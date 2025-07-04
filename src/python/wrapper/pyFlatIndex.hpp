#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>
#include "src/index/FlatIndex.hpp"

namespace py = pybind11;

class PyFlatIndex {
private:
    std::unique_ptr<FlatIndex> index_;
    
public:
    // 构造函数
    PyFlatIndex(uint64_t dim, uint64_t capacity, bool isFloat16 = false, 
                MetricType metricType = MetricType::METRIC_INNER_PRODUCT, kp::Manager* mgr = nullptr);
    PyFlatIndex(uint64_t dim, kp::Manager* mgr, MetricType metricType);
    
    // 向量操作
    void add_vectors(py::array_t<float> vectors);
    
    // 基本信息获取
    uint64_t get_num() const;
    uint64_t get_dim() const;
    uint64_t get_capacity() const;
    bool is_float16() const;
    
    // 查询方法
    py::tuple query(py::array_t<float> queries, uint64_t k, DeviceType device);
    py::tuple query_range(py::array_t<float> queries, uint64_t k, uint64_t start, uint64_t end, 
                         DeviceType device);
    
    // 最终实际不应该暴露device参数，应该在FlatIndex内部进行调度处理
    // TODO
    py::tuple search(py::array_t<float> queries, uint64_t k);

    // 重建向量
    py::array_t<float> reconstruct(uint64_t idx);
    
    // 文件操作
    int save(const std::string& filename);
    int load(const std::string& filename);
};