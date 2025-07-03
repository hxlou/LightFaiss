#include "src/index/FlatIndex.hpp"

#include <iostream>

void testFlatIndexSL() {
    // 测试一下 FlatIndex 的保存和从文件恢复的功能
    kp::Manager* mgr = nullptr; // 假设有一个Kompute管理器
    FlatIndex flatIndex(10, mgr);

    // 构造数据，每个数据为 i.0 ~i.9
    std::vector<float> data(100);
    for (uint64_t i = 0; i < 10; ++i) {
        for (uint64_t j = 0; j < 10; ++j) {
            data[i * 10 + j] = static_cast<float>(i + j * 0.1);
        }
    }

    // 添加数据到 FlatIndex
    flatIndex.addVector(data.data(), 10);

    // 保存到文件
    std::string filename = "data/flat_index_test.bin";
    int saveResult = flatIndex.save(filename);
    if (saveResult != 0) {
        std::cerr << "Failed to save FlatIndex to file: " << filename << std::endl;
        return;
    }

    // 从文件加载 FlatIndex
    FlatIndex loadedFlatIndex(10, mgr);
    int loadResult = loadedFlatIndex.load(filename);
    if (loadResult != 0) {
        std::cerr << "Failed to load FlatIndex from file: " << filename << std::endl;
        return;
    }
    // 验证加载的数据
    if (loadedFlatIndex.getNum() != flatIndex.getNum() ||
        loadedFlatIndex.getDim() != flatIndex.getDim() ||
        loadedFlatIndex.isFloat16() != flatIndex.isFloat16()) {
        std::cerr << "Loaded FlatIndex does not match original." << std::endl;
        return;
    }

    // 验证数据是否正确
    std::vector<float> reconstructedData(10);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            loadedFlatIndex.reconstruct(i, reconstructedData.data());
            if (reconstructedData[j] != data[i * 10 + j]) {
                std::cerr << "Data mismatch at index " << i << ", " << j
                          << ": expected " << data[i * 10 + j]
                          << ", got " << reconstructedData[j] << std::endl;
                return;
            }
        }
    }
    std::cout << "FlatIndex test passed successfully!" << std::endl;
};


int main() {
    testFlatIndexSL();
    // Add more tests or functionality as needed

    return 0;
}