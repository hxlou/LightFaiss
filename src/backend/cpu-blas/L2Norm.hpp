#pragma once

#include "src/index/MetricType.hpp"

#include <cstdint> // For uint, uint64_t, etc.
#include <cstddef> // For size_t
#include <memory>
#include <vector>
#include <algorithm>

namespace cpu_blas {
    void normalized_L2(
        size_t dim,
        size_t nx,
        float* x
    );

    void fvec_renorm_L2(
        size_t dim,
        size_t nx,
        float* x
    );

    void fvec_renorm_L2_noomp (
        size_t dim,
        size_t nx,
        float* x
    );

    void fvec_renorm_L2_omp (
        size_t dim,
        size_t nx,
        float* x
    );

}