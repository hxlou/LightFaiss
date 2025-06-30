#include <kompute/Kompute.hpp>

#include "backend/gpu-kompute/distance.hpp"
#include "backend/gpu-kompute/readShader.hpp"

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
    bool transX,
    bool transY,
    float* metricArg
) {
    if (metricType == METRIC_L2) {
        calL2(mgr, query, data, nQuery, nData, dim, k, distances, results, dataNorm, transX, transY);
    } else if (metricType == METRIC_INNER_PRODUCT) {
        calIP(mgr, query, data, nQuery, nData, dim, k, distances, results, dataNorm, transX, transY);
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
    const float* yNorm,
    bool transX,
    bool transY
) {


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
    const float* yNorm,
    bool transX,
    bool transY
) {
    // 计算内积距离，使用kompute接口
    std::shared_ptr<kp::TensorT<float>> IP = mgr->tensorT<float>(std::vector<float>(nx * ny, 0.0f));
    std::shared_ptr<kp::TensorT<float>> Tx  = mgr->tensorT<float>(std::vector<float>(x, x + nx * dim));
    std::shared_ptr<kp::TensorT<float>> Ty  = mgr->tensorT<float>(std::vector<float>(y, y + ny * dim));

    matmul(mgr, Tx, Ty, IP, nx, ny, dim, transX, transY);

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
    auto shader = readSpvFile("src/backend/gpu-kompute/shaders/matmul.comp.spv");

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

}