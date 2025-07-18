#include "backend/npu-hexagon/distance.hpp"

#include <memory>          // std::unique_ptr
#include <cstddef>         // size_t
#include <cmath>           // HUGE_VALF
#include <omp.h>           // OpenMP并行 (如果用OpenMP)
#include <queue>          // std::priority_queue
#include <utility>         // std::pair
#include <algorithm>       // std::min
#include <iostream>        // std::cout, std::endl
#include <string>
#include <ctime>
#include "backend/npu-hexagon/calculator-api.h"
#include <android/log.h>
#define LOG_TAG "MATMUL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)


#ifndef FINTEGER
#define FINTEGER long
#endif

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
    float* metricArg
) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);


    if (nQuery == 0 || k == 0 || dim == 0 || nData == 0)
        return;
    // 选择合适的计算内积的函数
    if (metricType == MetricType::METRIC_INNER_PRODUCT) {
        npu_hexagon::calIPHexagon(query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    } else {
        npu_hexagon::calL2Hexagon(query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    }
           // 记录结束时间
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    // 计算耗时（单位：毫秒）
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0 +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1000000.0;

    // 输出到安卓日志
    __android_log_print(ANDROID_LOG_INFO, "npu_hexagon", "query耗时: %.3f ms", elapsed);

}

void cpu_gemm_naive(
    const float* A, const float* B,
    size_t M, size_t K, size_t N,
    float* C,
    bool transA, bool transB)
{
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a = transA ? A[k * M + m] : A[m * K + k];
                float b = transB ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = sum;
        }
    }
}

void calculator_gemm_with_check_cpp(
    const float* A, const float* B,
    size_t M, size_t K, size_t N,
    float* C,
    bool transA, bool transB)
{
	memset(C, 0, sizeof(float) * M * N);

    std::vector<float> C_cpu(M * N, 0.0f);
    cpu_gemm_naive(A, B, M, K, N, C_cpu.data(), transA, transB);

	__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "=== CPU Reference Result Matrix (%zu x %zu) ===", M, N);
    for (size_t i = 0; i < M; ++i) {
        std::string row = "Row " + std::to_string(i) + ":";
        for (size_t j = 0; j < N; ++j) {
            row += " " + std::to_string(C_cpu[i * N + j]);
        }
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", row.c_str());
    }

    calculator_gemm_cpp(A, B, M, K, N, C, transA, transB);

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "=== NPU Result Matrix (%zu x %zu) ===", M, N);
    for (size_t i = 0; i < M; ++i) {
        std::string row = "Row " + std::to_string(i) + ":";
        for (size_t j = 0; j < N; ++j) {
            row += " " + std::to_string(C[i * N + j]);
        }
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", row.c_str());
    }

    float min_val = C[0], max_val = C[0], sum = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        min_val = std::min(min_val, C[i]);
        max_val = std::max(max_val, C[i]);
        sum += C[i];
    }
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Result stats - Min: %.6f, Max: %.6f, Mean: %.6f, Total elements: %zu", min_val, max_val, sum / (M * N), M * N);

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "=== NPU vs CPU Comparison ===");
    float max_diff = 0.0f, sum_diff = 0.0f;
    size_t count_significant = 0;
    const float threshold = 1e-5f;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::abs(C[i] - C_cpu[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > threshold) count_significant++;
    }
    float avg_diff = sum_diff / (M * N);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Max difference: %.6f", max_diff);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Average difference: %.6f", avg_diff);
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Elements with significant difference (>1e-5): %zu/%zu", count_significant, M * N);
    if (count_significant == 0) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "✓ NPU and CPU results match within tolerance");
    } else {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "✗ NPU and CPU results mismatch! Significant diffs found.");
    }
}

void calL2Hexagon(
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
    if (nx == 0 || ny == 0 || k == 0) {
        return;
    }

    std::unique_ptr<float[]> x_norms(new float[nx]);
    fvec_norms_L2sqr(x_norms.get(), x, dim, nx);

    std::unique_ptr<float[]> del2;
    if (!yNorm) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, dim, ny);
        yNorm = y_norms2;
    }

	// 内存限制: 12 MB
    const size_t MEM_LIMIT_BYTES = 12 * 1024 * 1024;
    const size_t MEM_LIMIT_FLOATS = MEM_LIMIT_BYTES / sizeof(float);

    size_t bs_x = 1024;
    if (bs_x > nx) bs_x = nx;

    size_t bs_y;
    if (bs_x * dim >= MEM_LIMIT_FLOATS) {
        bs_x = MEM_LIMIT_FLOATS / (dim + 1);
        if (bs_x == 0) bs_x = 1;
    }
    
    size_t remaining_mem = MEM_LIMIT_FLOATS - bs_x * dim;
    bs_y = remaining_mem / (dim + bs_x);
    if (bs_y == 0) bs_y = 1;
    if (bs_y > ny) bs_y = ny;

    #pragma omp parallel for
    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = std::min(nx, i0 + bs_x);
        size_t nxi = i1 - i0;

        using HeapElement = std::pair<float, uint64_t>;
        std::vector<std::priority_queue<HeapElement>> query_heaps(nxi);

        std::unique_ptr<float[]> ip_block(new float[nxi * bs_y]);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = std::min(ny, j0 + bs_y);
            size_t nyi = j1 - j0;

            calculator_gemm_cpp(
                x + i0 * dim,
                y + j0 * dim,
                nxi,
                dim,
                nyi,
                ip_block.get(),
                false,
                true
            );

            for (size_t i_local = 0; i_local < nxi; ++i_local) {
                size_t i_global = i0 + i_local;
                const float* ip_row = ip_block.get() + i_local * nyi;

                for (size_t j_local = 0; j_local < nyi; ++j_local) {
                    size_t j_global = j0 + j_local;
                    float ip = ip_row[j_local];

                    float d = x_norms[i_global] + yNorm[j_global] - 2 * ip;
                    d = std::max(0.0f, d);

                    if (query_heaps[i_local].size() < k) {
                        query_heaps[i_local].push({d, j_global});
                    } else if (d < query_heaps[i_local].top().first) {
                        query_heaps[i_local].pop();
                        query_heaps[i_local].push({d, j_global});
                    }
                }
            }
        }

        for (size_t i_local = 0; i_local < nxi; ++i_local) {
            size_t i_global = i0 + i_local;
            auto& heap = query_heaps[i_local];
            
            size_t current_k = heap.size();
            for (size_t j = 0; j < current_k; ++j) {
                const auto& [distance, index] = heap.top();
                size_t out_idx = i_global * k + (current_k - 1 - j);
                outDistances[out_idx] = distance;
                outIndices[out_idx] = index;
                heap.pop();
            }
        }
    }
}

void calIPHexagon(
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
    if (nx == 0 || ny == 0 || k == 0) {
        return;
    }

    // 内存限制: 12 MB
    const size_t MEM_LIMIT_BYTES = 12 * 1024 * 1024;
    const size_t MEM_LIMIT_FLOATS = MEM_LIMIT_BYTES / sizeof(float);

    size_t bs_x = 1024;
    if (bs_x > nx) bs_x = nx;

    size_t bs_y;
    if (bs_x * dim >= MEM_LIMIT_FLOATS) {
        bs_x = MEM_LIMIT_FLOATS / (dim + 1);
        if (bs_x == 0) bs_x = 1;
    }
    
    size_t remaining_mem = MEM_LIMIT_FLOATS - bs_x * dim;
    bs_y = remaining_mem / (dim + bs_x);
    if (bs_y == 0) bs_y = 1;
    if (bs_y > ny) bs_y = ny;

    #pragma omp parallel for
    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = std::min(nx, i0 + bs_x);
        size_t nxi = i1 - i0;

        using HeapElement = std::pair<float, uint64_t>;
        using MinHeap = std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<HeapElement>>;
        std::vector<MinHeap> query_heaps(nxi);

        std::unique_ptr<float[]> ip_block(new float[nxi * bs_y]);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = std::min(ny, j0 + bs_y);
            size_t nyi = j1 - j0;

            calculator_gemm_cpp(
                x + i0 * dim,
                y + j0 * dim,
                nxi,
                dim,
                nyi,
                ip_block.get(),
                false,
                true
            );

            for (size_t i_local = 0; i_local < nxi; ++i_local) {
                const float* ip_row = ip_block.get() + i_local * nyi;
                for (size_t j_local = 0; j_local < nyi; ++j_local) {
                    float ip = ip_row[j_local];
                    size_t j_global = j0 + j_local;

                    if (query_heaps[i_local].size() < k) {
                        query_heaps[i_local].push({ip, j_global});
                    } else if (ip > query_heaps[i_local].top().first) {
                        query_heaps[i_local].pop();
                        query_heaps[i_local].push({ip, j_global});
                    }
                }
            }
        }

        for (size_t i_local = 0; i_local < nxi; ++i_local) {
            size_t i_global = i0 + i_local;
            MinHeap& heap = query_heaps[i_local];
            
            size_t current_k = heap.size();
            for (size_t j = 0; j < current_k; ++j) {
                const auto& [ip_value, index] = heap.top();

                size_t out_idx = i_global * k + (current_k - 1 - j);
                outDistances[out_idx] = ip_value;
                outIndices[out_idx] = index;
                heap.pop();
            }
        }
    } // 结束对 x 块的并行遍历
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

} // namespace npu_hexagon