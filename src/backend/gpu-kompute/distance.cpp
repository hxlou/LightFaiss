#include <kompute/Kompute.hpp>

#include "backend/gpu-kompute/distance.hpp"
#include "backend/gpu-kompute/readShader.hpp"
#include "index/FlatIndex.hpp"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <vector>

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
    std::vector<uint32_t> shader;
    {
        std::lock_guard<std::mutex> lock(FlatIndex::assetManagerMutex_);
        shader = readSpvAsset(FlatIndex::assetManager_, "shaders/matmul.comp.spv");
    }

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

    auto algorithm = mgr->algorithm(
        memories,
        shader,
        kp::Workgroup({static_cast<uint32_t>(m), static_cast<uint32_t>(n), 1}),
        std::vector<float>{},
        pushConsts
    );

    auto seq = mgr->sequence()
        ->record<kp::OpSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({out});

    seq->eval();
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
    auto shader = readSpvFile("src/backend/gpu-kompute/shaders/L2Norm.comp.spv");

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

    auto shader = readSpvFile("src/backend/gpu-kompute/shaders/L2NormAdd.comp.spv");

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