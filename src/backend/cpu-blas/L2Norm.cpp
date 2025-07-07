#include "src/backend/cpu-blas/L2Norm.hpp"
#include "src/backend/cpu-blas/distance.hpp"

#include <cmath>

namespace cpu_blas {

void normalized_L2(
    size_t dim,
    size_t nx,
    float* x
) {
    fvec_renorm_L2(dim, nx, x);
}

void fvec_renorm_L2(
    size_t dim,
    size_t nx,
    float* x
) {
    if (nx == 0)
        return;
    if (nx > 10000) 
        fvec_renorm_L2_omp(dim, nx, x);
    else
        fvec_renorm_L2_noomp(dim, nx, x);
}

void fvec_renorm_L2_noomp (
    size_t dim,
    size_t nx,
    float* x
) {
    for (size_t i = 0; i < nx; ++i) {
        float* xi = x + i * dim;
        float norm = fvec_norm_L2sqr(xi, dim);
        if (norm > 0) {
            norm = 1.0f / sqrtf(norm);
            for (size_t j = 0; j < dim; ++j) {
                xi[j] *= norm;
            }
        }
    }
}

void fvec_renorm_L2_omp (
    size_t dim,
    size_t nx,
    float* x
) {
#pragma omp parallel for if (nx > 10000)
    for (size_t i = 0; i < nx; ++i) {
        float* xi = x + i * dim;
        float norm = fvec_norm_L2sqr(xi, dim);
        if (norm > 0) {
            norm = 1.0f / sqrtf(norm);
            for (size_t j = 0; j < dim; ++j) {
                xi[j] *= norm;
            }
        }
    }
}


}