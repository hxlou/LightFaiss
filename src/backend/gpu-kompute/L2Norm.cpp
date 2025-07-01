#include "src/backend/gpu-kompute/L2Norm.hpp"
#include "src/backend/gpu-kompute/distance.hpp"
#include <kompute/Kompute.hpp>

namespace gpu_kompute {

void normalized_L2(
    kp::Manager* mgr,
    size_t dim,
    size_t nx,
    float* x
) {
    fvec_renorm_L2(mgr, dim, nx, x);
}

void fvec_renorm_L2(
    kp::Manager* mgr,
    size_t dim,
    size_t nx,
    float* x
) {
    // 与vecsNorm类似，但是这个函数是原地进行修改，而不是进行额外的计算
    // TODO

}

}