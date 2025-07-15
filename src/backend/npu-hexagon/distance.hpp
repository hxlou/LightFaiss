// #include "src/index/FlatIndex.hpp"

#pragma once

#include "index/MetricType.hpp"

#include <cstdint> // For uint, uint64_t, etc.
#include <cstddef> // For size_t
#include <memory>
#include <vector>
#include <algorithm>

namespace npu_hexagon {

void query(
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

void calL2Hexagon(
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

void fvec_norms_L2sqr (
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx);

float fvec_norm_L2sqr(const float* x, size_t d);

void calIPHexagon(
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

}