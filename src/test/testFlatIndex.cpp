#include "src/index/FlatIndex.hpp"
#include "src/backend/cpu-blas/L2Norm.hpp"

#include <vector>
#include <iostream>

void testFlatIndexCpuL2() {
    FlatIndex index(2, 1000, false, MetricType::METRIC_L2, nullptr);

    // 添加10000个向量
    std::vector<float> vecs(2 * 10000, 1.0f); // 创建10000个全1的向量
    for (int i = 0; i < 10000; ++i) {
        for (int j = 0; j < 2; ++j) {
            vecs[i * 2 + j] = static_cast<float>(i * 0.01); // 示例数据
        }
    }

    index.addVector(vecs.data(), 10000);

    std::cout << "Adding 10000 vectors to the index..." << std::endl;
    std::cout << "Vector size: " << vecs.size() << std::endl;
    std::cout << "Vector capacity: " << index.getCapacity() << std::endl;
    std::cout << "Vector dimension: " << index.getDim() << std::endl;
    std::cout << "Vector number: " << index.getNum() << std::endl;

    int nQuery = 100;

    // 新建nQuery个查询向量
    std::vector<float> queries(2 * nQuery, 1.0f); // 创建nQuery个全1的查询向量
    for (int i = 0; i < nQuery; ++i) {
        for (int j = 0; j < 2; ++j) {
            queries[i * 2 + j] = static_cast<float>(i * 0.1145); // 示例数据
        }
    }

    // 查询前5个匹配的向量
    std::vector<uint64_t> results(5 * nQuery); // 存储查询结果
    std::vector<float> distances(5 * nQuery);
    index.query(5, 0, 10000, DeviceType::CPU_BLAS, nQuery, queries.data(), results.data(), distances.data());

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

void testFlatIndexCpuIP() {
    FlatIndex index(2, 1000, true, MetricType::METRIC_INNER_PRODUCT, nullptr);

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
    index.query(5, 0, 1000, DeviceType::CPU_BLAS, nQuery, queries.data(), results.data(), distances.data());

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

void FlatIndexCpuRenorm() {
    // 不用查询，单单测试一下remorm函数是否正常工作
    // 新建向量 1000 个，把半径为10的圆周分成1000份
    int nData = 1000;
    std::vector<float> vecs(2 * nData);
    for (int i = 0; i < nData; ++i) {
        vecs[i * 2] = cos(2 * M_PI * i / nData) * 10.0f;
        vecs[i * 2 + 1] = sin(2 * M_PI * i / nData) * 10.0f;
    }

    cpu_blas::normalized_L2(2, nData, vecs.data());

    bool isPassed = true;
    // 检测归一化后的向量是否正确
    for (int i = 0; i < nData; ++i) {
        // 理论上现在的vecs应该是把半径为1的圆周分成1000份
        float x = vecs[i * 2];
        float y = vecs[i * 2 + 1];
        float gx = cos(2 * M_PI * i / nData);
        float gy = sin(2 * M_PI * i / nData);
        if (std::abs(x - gx) > 1e-5 || std::abs(y - gy) > 1e-5) {
            std::cout << "Renormalization failed at index " << i 
                      << ": expected (" << gx << ", " << gy << "), got (" 
                      << x << ", " << y << ")" << std::endl;
            isPassed = false;
        }
    }

    if (isPassed) {
        std::cout << "Renormalization test passed!" << std::endl;
    } else {
        std::cout << "Renormalization test failed!" << std::endl;
    }
}

int main () {

    testFlatIndexCpuL2();
    std::cout << "-------------------------" << std::endl;
    testFlatIndexCpuIP();
    std::cout << "-------------------------" << std::endl;
    FlatIndexCpuRenorm();

    return 0;
}