#include <kompute/Kompute.hpp>

#include "backend/gpu-kompute/distance.hpp"
#include "backend/gpu-kompute/readShader.hpp"
#include "backend/gpu-kompute/shader.hpp"
#include "index/FlatIndex.hpp"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <vector>
#include <sstream>
#include <algorithm>
#include <numeric>

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
    float* metricArg
) {
    if (metricType == METRIC_L2) {
        calL2(mgr, query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    } else if (metricType == METRIC_INNER_PRODUCT) {
        calIP(mgr, query, data, nQuery, nData, dim, k, distances, results, dataNorm);
    } else {
        // 其他距离计算方式可以在这里添加
        // throw std::invalid_argument("Unsupported metric type");
    }
}

/*
    使用BLAS计算L2距离
*/
void calL2(
    kp::Manager* mgr,
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
    // xNorm yNorm IP
    std::shared_ptr<kp::TensorT<float>> X = mgr->tensorT<float>(std::vector<float>(x, x + nx * dim));
    std::shared_ptr<kp::TensorT<float>> Y = mgr->tensorT<float>(std::vector<float>(y, y + ny * dim));
    std::shared_ptr<kp::TensorT<float>> IP = mgr->tensorT<float>(std::vector<float>(nx * ny, 0.0f));
    std::shared_ptr<kp::TensorT<float>> XNorm = mgr->tensorT<float>(std::vector<float>(nx, 0.0f));
    std::shared_ptr<kp::TensorT<float>> YNorm;
    std::shared_ptr<kp::TensorT<float>> L2 = mgr->tensorT<float>(std::vector<float>(nx * ny, 0.0f));

    // YNorm
    if (yNorm != nullptr) {
        YNorm = mgr->tensorT<float>(std::vector<float>(yNorm, yNorm + ny));
    } else {
        YNorm = mgr->tensorT<float>(std::vector<float>(ny, 0.0f));
        vecsNorm(mgr, Y, YNorm, ny, dim);
    }

    // XNorm
    vecsNorm(mgr, X, XNorm, nx, dim);

    // IP
    matmul(mgr, X, Y, IP, nx, ny, dim, false, true);

    // 计算L2距离
    calL2Add(mgr, XNorm, YNorm, IP, L2, nx, ny);

    // 从L2中排序并赋值结果
    std::vector<std::pair<float, uint64_t>> results(ny);
    for (uint64_t i = 0; i < nx; ++i) {
        for (uint64_t j = 0; j < ny; ++j) {
            float value = L2->data()[i * ny + j];
            results[j] = std::make_pair(value, j); // 存储距离和索引
        }
        std::sort(results.begin(), results.end(),
                          [](const std::pair<float, uint64_t>& a, const std::pair<float, uint64_t>& b) {
                              return a.first < b.first; // 升序排序
                          });
        // 将前k个结果写入输出
        for (uint64_t j = 0; j < k; ++j) {
            outDistances[i * k + j] = results[j].first;
            outIndices[i * k + j] = results[j].second;
        }
    }

}

/*
    使用BLAS计算IP距离
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
    const float* yNorm
) {
    // 计算内积距离，使用kompute接口
    std::shared_ptr<kp::TensorT<float>> IP = mgr->tensorT<float>(std::vector<float>(nx * ny, 0.0f));
    std::shared_ptr<kp::TensorT<float>> Tx  = mgr->tensorT<float>(std::vector<float>(x, x + nx * dim));
    std::shared_ptr<kp::TensorT<float>> Ty  = mgr->tensorT<float>(std::vector<float>(y, y + ny * dim));

    matmul(mgr, Tx, Ty, IP, nx, ny, dim, false, true);

    // 从IP中复制结果
    std::vector<std::pair<float, uint64_t>> results(ny);
    for (uint64_t i = 0; i < nx; ++i) {
        
        for (uint64_t j = 0; j < ny; ++j) {
            float value = IP->data()[i * ny + j];
            results[j] = std::make_pair(value, j); // 存储距离和索引
        }

        std::sort(results.begin(), results.end(),
                          [](const std::pair<float, uint64_t>& a, const std::pair<float, uint64_t>& b) {
                              return a.first > b.first; // 降序排序
                          });

        // 将前k个结果写入输出
        for (uint64_t j = 0; j < k; ++j) {
            outDistances[i * k + j] = results[j].first;
            outIndices[i * k + j] = results[j].second;
        }
    }

    return;
}

void matmul (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> x,
    std::shared_ptr<kp::TensorT<float>> y,
    std::shared_ptr<kp::TensorT<float>> out,
    size_t m,
    size_t n,
    size_t k,
    bool transX,
    bool transY
) {
    // 输出矩阵维度信息
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "Matrix dimensions: m=%zu, n=%zu, k=%zu, transX=%d, transY=%d", 
                       m, n, k, transX, transY);

	const auto& xData = x->data();
	const auto& yData = y->data();

    // 输出完整的矩阵A
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "=== Matrix A (%zu x %zu) ===", m, k);
    for (size_t i = 0; i < m; ++i) {
        std::stringstream ss;
        ss << "Row " << i << ": ";
        for (size_t j = 0; j < k; ++j) {
            size_t idx = transX ? (j * m + i) : (i * k + j);
            ss << xData[idx] << " ";
        }
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "%s", ss.str().c_str());
    }
    
    // 输出完整的矩阵B
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "=== Matrix B (%zu x %zu) ===", k, n);
    for (size_t i = 0; i < k; ++i) {
        std::stringstream ss;
        ss << "Row " << i << ": ";
        for (size_t j = 0; j < n; ++j) {
            size_t idx = transY ? (j * k + i) : (i * n + j);
            ss << yData[idx] << " ";
        }
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "%s", ss.str().c_str());
    }

    uint32_t* shaderPtr = reinterpret_cast<uint32_t*>(gpu_kompute::matmul_o2_comp_spv);
    std::vector<uint32_t> shader(shaderPtr, shaderPtr + gpu_kompute::matmul_o2_comp_spv_len / sizeof(uint32_t));
    

    std::vector<uint32_t> pushConsts = {
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(k),
        static_cast<uint32_t>(transX),
        static_cast<uint32_t>(transY)
    };

    std::vector<std::shared_ptr<kp::Memory>> memories = {
        std::static_pointer_cast<kp::Memory>(x),
        std::static_pointer_cast<kp::Memory>(y),
        std::static_pointer_cast<kp::Memory>(out)
    };

	const uint32_t TILE_SIZE = 16;
	uint32_t groupCountX = (static_cast<uint32_t>(n) + TILE_SIZE - 1) / TILE_SIZE;
	uint32_t groupCountY = (static_cast<uint32_t>(m) + TILE_SIZE - 1) / TILE_SIZE;

	auto algorithm = mgr->algorithm(
		memories,
		shader,
		kp::Workgroup({ groupCountX, groupCountY, 1 }),
		std::vector<float>{},
		pushConsts
	);

    auto seq = mgr->sequence()
        ->record<kp::OpSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({out});

    seq->eval();
    
    // 输出完整的结果矩阵
    const auto& outData = out->data();
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "=== Result Matrix (%zu x %zu) ===", m, n);
    for (size_t i = 0; i < m; ++i) {
        std::stringstream ss;
        ss << "Row " << i << ": ";
        for (size_t j = 0; j < n; ++j) {
            size_t idx = i * n + j;
            ss << outData[idx] << " ";
        }
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "%s", ss.str().c_str());
    }
    
    // 输出结果矩阵的统计信息
    size_t totalElements = m * n;
    if (totalElements > 0) {
        // 直接使用outData作为指针
        const float* dataPtr = &outData[0];
        float minVal = *std::min_element(dataPtr, dataPtr + totalElements);
        float maxVal = *std::max_element(dataPtr, dataPtr + totalElements);
        float sum = std::accumulate(dataPtr, dataPtr + totalElements, 0.0f);
        float mean = sum / totalElements;
        
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "Result stats - Min: %f, Max: %f, Mean: %f, Total elements: %zu", 
                           minVal, maxVal, mean, totalElements);
    }
	    
    // 添加朴素的C++矩阵乘法验证
    std::vector<float> cpuResult(m * n, 0.0f);
    
    // CPU朴素矩阵乘法计算
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                // 考虑转置情况
                size_t xIdx = transX ? (l * m + i) : (i * k + l);
                size_t yIdx = transY ? (j * k + l) : (l * n + j);
                sum += xData[xIdx] * yData[yIdx];
            }
            cpuResult[i * n + j] = sum;
        }
    }
    
    // 输出CPU计算结果矩阵
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "=== CPU Reference Result Matrix (%zu x %zu) ===", m, n);
    for (size_t i = 0; i < m; ++i) {
        std::stringstream ss;
        ss << "Row " << i << ": ";
        for (size_t j = 0; j < n; ++j) {
            size_t idx = i * n + j;
            ss << cpuResult[idx] << " ";
        }
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "%s", ss.str().c_str());
    }
    
    // 比较GPU和CPU结果的差异
    float maxDiff = 0.0f;
    float totalDiff = 0.0f;
    size_t diffCount = 0;
    
    for (size_t i = 0; i < totalElements; ++i) {
        float diff = std::abs(outData[i] - cpuResult[i]);
        if (diff > 1e-5f) {  // 设置一个小的阈值
            diffCount++;
        }
        maxDiff = std::max(maxDiff, diff);
        totalDiff += diff;
    }
    
    float avgDiff = totalElements > 0 ? totalDiff / totalElements : 0.0f;
    
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "=== GPU vs CPU Comparison ===");
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "Max difference: %f", maxDiff);
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "Average difference: %f", avgDiff);
    __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "Elements with significant difference (>1e-5): %zu/%zu", diffCount, totalElements);
    
    if (maxDiff < 1e-4f) {
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "✓ GPU and CPU results match within tolerance");
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "MATMUL", "⚠ GPU and CPU results differ significantly!");
    }
}

void vecsNorm (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> vecs,
    std::shared_ptr<kp::TensorT<float>> norms,
    size_t n,
    size_t dim
) {
    // vecs  : n * dim
    // norms : n * 1
    // norms[i] = vecs[i*dim + 0] ^ 2 + ... + vecs[i * dim + (dim - 1)] ^ 2
    // auto shader = readSpvFile("src/backend/gpu-kompute/shaders/L2Norm.comp.spv");
    uint32_t* shaderPtr = reinterpret_cast<uint32_t*>(gpu_kompute::L2Norm_comp_spv);
    std::vector<uint32_t> shader(shaderPtr, shaderPtr + gpu_kompute::L2Norm_comp_spv_len / sizeof(uint32_t));

    std::vector<uint32_t> pushConsts = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(dim)
    };

    std::vector<std::shared_ptr<kp::Memory>> memories = {
        std::static_pointer_cast<kp::Memory>(vecs),
        std::static_pointer_cast<kp::Memory>(norms)
    };

    auto algorithm = mgr->algorithm(
        memories,
        shader,
        kp::Workgroup({static_cast<uint32_t>(n), 1, 1}),
        std::vector<float>{},
        pushConsts
    );

    auto seq = mgr->sequence()
        ->record<kp::OpSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({norms});

    seq->eval();
}

void calL2Add (
    kp::Manager* mgr,
    std::shared_ptr<kp::TensorT<float>> xNorm,      // 1 * nx
    std::shared_ptr<kp::TensorT<float>> yNorm,      // 1 * ny
    std::shared_ptr<kp::TensorT<float>> IP,         // nx * ny
    std::shared_ptr<kp::TensorT<float>> L2,         // nx * ny
    size_t nx,
    size_t ny
) {
    // L2 = xNorm + yNorm - 2 * IP
    // L2[i][j] = xNorm[i] + yNorm[j] - 2 * IP[i][j]

    // auto shader = readSpvFile("src/backend/gpu-kompute/shaders/L2NormAdd.comp.spv");
    uint32_t* shaderPtr = reinterpret_cast<uint32_t*>(gpu_kompute::L2NormAdd_comp_spv);
    std::vector<uint32_t> shader(shaderPtr, shaderPtr + gpu_kompute::L2NormAdd_comp_spv_len / sizeof(uint32_t));

    std::vector<uint32_t> pushConsts = {
        static_cast<uint32_t>(nx),
        static_cast<uint32_t>(ny)
    };

    std::vector<std::shared_ptr<kp::Memory>> memories = {
        std::static_pointer_cast<kp::Memory>(xNorm),
        std::static_pointer_cast<kp::Memory>(yNorm),
        std::static_pointer_cast<kp::Memory>(IP),
        std::static_pointer_cast<kp::Memory>(L2)
    };

    auto algorithm = mgr->algorithm(
        memories,
        shader,
        kp::Workgroup({static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), 1}),
        std::vector<float>{},
        pushConsts
    );

    auto seq = mgr->sequence()
        ->record<kp::OpSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({L2});

    seq->eval();
}

}