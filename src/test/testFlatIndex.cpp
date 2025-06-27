#include "index/FlatIndex.hpp"

#include <vector>
#include <iostream>


int main () {
    // 创建一个FlatIndex实例
    FlatIndex index(128, 1000, false, MetricType::METRIC_INNER_PRODUCT, nullptr);

    // 添加一些向量
    std::vector<float> vecs(128 * 10, 1.0f); // 创建10个全1的向量
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 128; ++j) {
            vecs[i * 128 + j] = static_cast<float>(i); // 示例数据
        }
    }

    // 添加向量到索引中
    index.addVector(vecs.data(), 10);

    std::cout << "After Insert" << std::endl;

    // 查询向量
    std::vector<float> query(128, 1.0f); // 创建一个全1的查询向量
    std::vector<uint64_t> results(5);
    std::vector<float> distances(5);
    index.query(5, 0, 1000, DeviceType::CPU_BLAS, 1, query.data(), results.data(), distances.data());

    std::cout << "After Query" << std::endl;
    // 输出查询结果
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Result " << i << ": Index = " << results[i] << ", Distance = " << distances[i] << std::endl;
    }

    return 0;


}