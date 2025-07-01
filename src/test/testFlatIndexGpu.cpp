#include "src/backend/gpu-kompute/distance.hpp"
#include "src/index/FlatIndex.hpp"

#include <vector>
#include <iostream>

void testFlatIndexGpuIP(kp::Manager* mgr) {
    FlatIndex index(2, 1000, true, MetricType::METRIC_INNER_PRODUCT, mgr);

    // 添加nData个向量，把圆周平分nData份，每个向量长度为1
    int nData = 1000;
    std::vector<float> vecs(2 * nData);
    for (int i = 0; i < nData; ++i) {
        vecs[i * 2] = cos(2 * M_PI * i / nData);
        vecs[i * 2 + 1] = sin(2 * M_PI * i / nData);
    }
    index.addVector(vecs.data(), nData);

    int nQuery = 10;
    std::vector<float> queries(2 * nQuery);
    for (int i = 0; i < nQuery; ++i) {
        queries[i * 2] = cos(2 * M_PI * i / nQuery);
        queries[i * 2 + 1] = sin(2 * M_PI * i / nQuery);
    }

    // 查询前k个匹配的向量
    int k = 5;
    std::vector<uint64_t> results(k * nQuery); // 存储查询结果
    std::vector<float> distances(k * nQuery);
    
    index.query(k, 0, nData, DeviceType::GPU_KOMPUTE, nQuery, 
                queries.data(), results.data(), distances.data());

    // 每个查询结果输出一次
    for (size_t i = 0; i < nQuery; i += 2) {
        std::cout << "Query " << i << "  " << queries[i * 2] << ", " << queries[i * 2 + 1] << std::endl;
        for (int j = 0; j < 3; ++j) {
            std::cout << "Result " << j << ": Index = " << results[i * k + j]
                      << ", Distance = " << distances[i * k + j] << "  ";
            // 重建向量
            std::vector<float> vec(2);
            index.reconstruct(results[i * k + j], vec.data());
            std::cout << "Reconstructed Vector: " << vec[0] << ", " << vec[1] << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    kp::Manager mgr;
    testFlatIndexGpuIP(&mgr);
    return 0;
}   