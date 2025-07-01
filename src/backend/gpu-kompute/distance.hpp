// #include "src/index/FlatIndex.hpp"

#pragma once

#include "src/index/MetricType.hpp"

#include <kompute/Kompute.hpp> // Assuming this is the correct path for the Kompute library

#include <cstdint> // For uint32_t, uint64_t, etc.
#include <cstddef> // For size_t

namespace gpu_kompute {

void query(
    kp::Manager* mgr,           // Kompute管理器
    uint64_t nQuery,
    uint64_t nData,
    uint64_t k,
    uint64_t dim,
    const float* query,
    const float* data,
    const float* dataNorm,
    float* distances,
    uint64_t* results,
    MetricType metricType,
    float* metricArg = nullptr
);

/*
    使用shader计算L2距离
*/
void calL2(
    kp::Manager* mgr,           // Kompute管理器
    const float* x,
    const float* y,
    size_t nx,
    size_t ny,
    size_t dim,
    uint64_t k,
    float* outDistances,
    uint64_t* outIndices,
    const float* yNorm = nullptr
);

/*
    使用shader计算IP距离
*/

void calIP(
    kp::Manager* mgr,           // Kompute管理器
    const float* x,             // 查询向量
    const float* y,             // 数据向量
    size_t nx,
    size_t ny,
    size_t dim,
    size_t k,
    float* outDistances,
    uint64_t* outIndices,
    const float* yNorm = nullptr
);

void matmul (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> x,
    std::shared_ptr<kp::TensorT<float>> y,
    std::shared_ptr<kp::TensorT<float>> out,
    size_t m,
    size_t n,
    size_t k,
    bool transX = false,
    bool transY = false
);

void vecsNorm (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> vecs,
    std::shared_ptr<kp::TensorT<float>> norms,
    size_t n,
    size_t dim
);

void calL2Add (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> xNorm,
    std::shared_ptr<kp::TensorT<float>> yNorm,
    std::shared_ptr<kp::TensorT<float>> IP,
    std::shared_ptr<kp::TensorT<float>> L2,
    size_t nx,
    size_t ny
);

}