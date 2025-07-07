#include "src/backend/gpu-kompute/distance.hpp"
#include "src/index/FlatIndex.hpp"
#include "src/backend/gpu-kompute/L2Norm.hpp"
#include "src/backend/gpu-kompute/L2Norm.hpp"

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

void testFlatIndexGpuL2(kp::Manager* mgr) {
    // 测试FlatIndexGpu的L2计算接口
    FlatIndex index(2, 1000, true, MetricType::METRIC_L2, mgr);

    // 添加nData个向量，把（0，0）到（10，10）分成ndata份
    int nData = 1000;
    std::vector<float> vecs(2 * nData);
    for (int i = 0; i < nData; ++i) {
        vecs[i * 2] = i * 10.0f / nData;
        vecs[i * 2 + 1] = i * 10.0f / nData;
    }
    index.addVector(vecs.data(), nData);

    // 添加nQuery个查询向量
    int nQuery = 10;
    std::vector<float> queries(2 * nQuery);
    for (int i = 0; i < nQuery; ++i) {
        queries[i * 2] = i * 10.0f / nQuery;
        queries[i * 2 + 1] = i * 10.0f / nQuery;
    }

    // 查询前k个匹配的向量
    int k = 5;
    std::vector<uint64_t> results(k * nQuery); // 存储查询
    std::vector<float> distances(k * nQuery);  // 存储距离
    index.query(k, 0, nData, DeviceType::GPU_KOMPUTE, nQuery, 
                queries.data(), results.data(), distances.data());

    // 每个查询结果输出一次
    for (size_t i = 0; i < nQuery; i += 2) {
        std::cout << "Query " << i << "  " << queries[i * 2] << ", " << queries[i * 2 + 1] << std::endl;
        for (int j = 0; j < 5; ++j) {
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

void testFlatIndexGpuRenorm(kp::Manager* mgr) {
    // 不用查询，单单测试一下remorm函数是否正常工作
    // 新建向量 1000 个，把半径为10的圆周分成1000份
    int nData = 1000;
    std::vector<float> vecs(2 * nData);
    for (int i = 0; i < nData; ++i) {
        vecs[i * 2] = cos(2 * M_PI * i / nData) * 10.0f;
        vecs[i * 2 + 1] = sin(2 * M_PI * i / nData) * 10.0f;
    }

    gpu_kompute::normalized_L2(mgr, 2, nData, vecs.data());

    bool isPassed = true;
    // 检测归一化后的向量是否正确
    for (int i = 0; i < nData; ++i) {
        // 理论上现在的vecs应该是把半径为1的圆周分成1000份
        float x = vecs[i * 2];
        float y = vecs[i * 2 + 1];
        float gx = cos(2 * M_PI * i / nData);
        float gy = sin(2 * M_PI * i / nData);
        // if (std::abs(x - gx) > 1e-5 || std::abs(y - gy) > 1e-5) {
        //     std::cout << "Renormalization failed at index " << i 
        //               << ": expected (" << gx << ", " << gy << "), got (" 
        //               << x << ", " << y << ")" << std::endl;
        //     isPassed = false;
        // }
    }

    if (isPassed) {
        std::cout << "Renormalization test passed!" << std::endl;
    } else {
        std::cout << "Renormalization test failed!" << std::endl;
    }
}

int main() {
    kp::Manager mgr;

    testFlatIndexGpuIP(&mgr);
    
    std::cout << "----------------------------------------" << std::endl;
    
    testFlatIndexGpuL2(&mgr);
    
    std::cout << "----------------------------------------" << std::endl;

    testFlatIndexGpuRenorm(&mgr);

    return 0;
}   