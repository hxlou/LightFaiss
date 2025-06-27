#include "backend/cpu-blas/distance.hpp"

#include <memory>          // std::unique_ptr
#include <cstddef>         // size_t
#include <cmath>           // HUGE_VALF
#include <omp.h>           // OpenMP并行 (如果用OpenMP)
#include <queue>          // std::priority_queue
#include <utility>         // std::pair
#include <algorithm>       // std::min
#include <iostream>        // std::cout, std::endl

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace cpu_blas {

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
    float* metricArg
) {
    if (nQuery == 0 || k == 0 || dim == 0 || nData == 0)
        return;
    // 选择合适的计算内积的函数
    if (metricType == MetricType::METRIC_INNER_PRODUCT) {
        cpu_blas::calIPBLAS(query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    } else {
        cpu_blas::calL2BLAS(query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    }
}

void calL2BLAS(
    const float* x,
    const float* y,
    size_t nx,
    size_t ny,
    size_t dim,
    uint64_t k,
    float* outDistances,
    uint64_t* outIndices,
    const float* yNorm
) {
    if (nx == 0 || ny == 0)
        return;

    // block size
    const size_t bs_x = 4096;
    const size_t bs_y = 1024;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    // 临时储存中间结果
    std::vector<std::pair<float, uint>> tmpSort(nx * (k + bs_y));

    // 计算x的范数
    fvec_norms_L2sqr(x_norms.get(), x, dim, nx);

    if (!yNorm) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, dim, ny);
        yNorm = y_norms2;
    }

    // 计算最终距离，外循环每次移动bs_x个元素，内循环每次移动bs_y个元素
    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = std::min(nx, i0 + bs_x);
        
        // 数据库分块
        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = std::min(ny, j0 + bs_y);

            // 计算实际的内积大小
            float one = 1, zero = 0;
            FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = dim;
            sgemm_(
                "Transpose",
                "Not transpose",
                &nyi,
                &nxi,
                &di,
                &one,
                y + j0 * dim,  // y的起始地址
                &di,
                x + i0 * dim,  // x的起始地址
                &di,
                &zero,
                ip_block.get(), // 存储结果的内存
                &nyi
            );

            // 最终处理
            for (int64_t i = i0; i < i1; ++i) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);
                
                for (size_t j = j0; j < j1; ++j) {
                    float ip = ip_line[j - j0];
                    float d = x_norms[i] + yNorm[j] - 2 * ip;

                    if (d < 0)
                        d = 0;

                    tmpSort[i * (k + bs_y) + k + j - j0] = std::make_pair(d, j);
                }
                // 对每个i的结果进行排序
                std::partial_sort(
                    tmpSort.begin() + i * (k + bs_y),
                    tmpSort.begin() + i * (k + bs_y) + k,
                    tmpSort.begin() + (i + 1) * (k + bs_y)
                );
            }
        }
    }

    // 将结果写入输出
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < k; ++j) {
            outDistances[i * k + j] = tmpSort[i * (k + bs_y) + j].first;
            outIndices[i * k + j] = tmpSort[i * (k + bs_y) + j].second;
        }
    }
} 

void calIPBLAS(
    const float* x,
    const float* y,
    size_t nx,
    size_t ny,
    size_t dim,
    uint64_t k,
    float* outDistances,
    uint64_t* outIndices,
    const float* yNorm
) {
    if (nx == 0 || ny == 0)
        return;

    const size_t bs_x = 4096;
    const size_t bs_y = 1024;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

    // 临时储存中间结果
    std::vector<std::pair<float, uint>> tmpSort(nx * (k + bs_y), std::make_pair(0.0, 1));

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = std::min(nx, i0 + bs_x);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = std::min(ny, j0 + bs_y);

            // 计算内积
            float one = 1, zero = 0;
            FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = dim;
            sgemm_(
                "Transpose",
                "Not transpose",
                &nyi,
                &nxi,
                &di,
                &one,
                y + j0 * dim,  // y的起始地址
                &di,
                x + i0 * dim,  // x的起始地址
                &di,
                &zero,
                ip_block.get(), // 存储结果的内存
                &nyi
            );
            for (int64_t i = i0; i < i1; ++i) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; ++j) {
                    tmpSort[i * (k + bs_y) + k + j - j0] = std::make_pair(ip_line[j - j0], j);
                }
                // 对每个i的结果进行排序
                std::partial_sort(
                    tmpSort.begin() + i * (k + bs_y),
                    tmpSort.begin() + i * (k + bs_y) + k,
                    tmpSort.begin() + (i + 1) * (k + bs_y),
                    [](const std::pair<float, uint64_t>& a, const std::pair<float, uint64_t>& b) {
                        return a.first > b.first; // 注意这里是 >，降序
                    }
                );
            }
        }
    }
    // 将结果写入输出
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < k; ++j) {
            outDistances[i * k + j] = tmpSort[i * (k + bs_y) + j].first;
            outIndices[i * k + j] = tmpSort[i * (k + bs_y) + j].second;
        }
    }

}

void fvec_norms_L2sqr (
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for if (nx > 10000)
    for (int64_t i = 0; i < nx; i++)
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}

} // namespace cpu_blas