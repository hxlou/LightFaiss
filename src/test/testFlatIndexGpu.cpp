#include "backend/gpu-kompute/distance.hpp"
#include "index/FlatIndex.hpp"

#include <vector>
#include <iostream>

void testFlatIndexGpuIP(kp::Manager* mgr) {
    FlatIndex index(2, 1000, true, MetricType::METRIC_INNER_PRODUCT, mgr);

    // 添加1000个向量，把圆周平分1000份，每个向量长度为1
    std::vector<float> vecs(2 * 1000);
    for (int i = 0; i < 1000; ++i) {
        vecs[i * 2] = cos(2 * M_PI * i / 1000);
        vecs[i * 2 + 1] = sin(2 * M_PI * i / 1000);
    }
    index.addVector(vecs.data(), 1000);

    std::cout << "Adding 1000 vectors to the index..." << std::endl;
    std::cout << "Vector size: " << vecs.size() << std::endl;
    std::cout << "Vector capacity: " << index.getCapacity() << std::endl;
    std::cout << "Vector dimension: " << index.getDim() << std::endl;
    std::cout << "Vector number: " << index.getNum() << std::endl;

    int nQuery = 100;
    
    // 新建nQuery个查询向量，把圆周评分nQuery份，每个向量长度为1
    std::vector<float> queries(2 * nQuery);
    for (int i = 0; i < nQuery; ++i) {
        queries[i * 2] = cos(2 * M_PI * i / nQuery);
        queries[i * 2 + 1] = sin(2 * M_PI * i / nQuery);
    }

    // 查询前5个匹配的向量
    std::vector<uint64_t> results(5 * nQuery); // 存储查询结果
    std::vector<float> distances(5 * nQuery);
    index.query(5, 0, 1000, DeviceType::GPU_KOMPUTE, nQuery, queries.data(), results.data(), distances.data());

    // 每个查询结果输出一次
    for (size_t i = 0; i < nQuery; i += nQuery / 5) {
        std::cout << "Query " << i << "  " << queries[i * 2] << ", " << queries[i * 2 + 1] << std::endl;
        for (int j = 0; j < 5; ++j) {
            std::cout << "Result " << j << ": Index = " << results[i * 5 + j]
                      << ", Distance = " << distances[i * 5 + j] << "  ";
            // 重建向量
            std::vector<float> vec(2);
            index.reconstruct(results[i * 5 + j], vec.data());
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