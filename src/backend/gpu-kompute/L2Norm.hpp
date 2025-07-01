#pragma once

#include "src/index/MetricType.hpp"

#include <kompute/Kompute.hpp> // Assuming this is the correct path for the Kompute library

#include <cstdint> // For uint32_t, uint64_t, etc.
#include <cstddef> // For size_t

namespace gpu_kompute {

void normalized_L2(
    kp::Manager* mgr,
    size_t dim,
    size_t nx,
    float* x
);

void fvec_renorm_L2(
    kp::Manager* mgr,
    size_t dim,
    size_t nx,
    float* x
);
    
}