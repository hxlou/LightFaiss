#include "backend/gpu-kompute/readShader.hpp"

#include <vector>
#include <string>
#include <fstream>  // 用于文件流操作
#include <iostream> // 用于错误输出 (你可以替换为你项目中的日志系统)
#include <cstdint>  // 用于 uint32_t

// #include <android/asset_manager.h>
// #include <android/asset_manager_jni.h>
#include <mutex>

namespace gpu_kompute {

std::vector<uint32_t> readSpvFile(const std::string& filename) {
    // 以二进制模式打开文件，并且初始定位到文件末尾 (std::ios::ate) 以便获取文件大小
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        // YOUR_LOG_ERROR("错误: 无法打开文件: %s", filename.c_str());
        std::cerr << "错误: 无法打开文件: " << filename << std::endl;
        return {}; // 返回空vector表示失败
    }

    // 获取文件大小 (因为定位在末尾，tellg() 返回的就是文件大小)
    std::streamsize fileSize = file.tellg();

    // 检查文件大小是否有效
    if (fileSize <= 0) {
        // YOUR_LOG_ERROR("错误: 文件 '%s' 为空或大小无效。", filename.c_str());
        std::cerr << "错误: 文件 '" << filename << "' 为空或大小无效。" << std::endl;
        file.close();
        return {};
    }

    // SPIR-V 文件大小必须是 uint32_t (4字节) 的整数倍
    if (fileSize % sizeof(uint32_t) != 0) {
        // YOUR_LOG_ERROR("错误: 文件 '%s' 的大小 (%lld字节) 不是 uint32_t (4字节) 的整数倍。",
        //                filename.c_str(), static_cast<long long>(fileSize));
        std::cerr << "错误: 文件 '" << filename << "' 的大小 (" << fileSize
                  << "字节) 不是 uint32_t (" << sizeof(uint32_t)
                  << "字节) 的整数倍。" << std::endl;
        file.close();
        return {};
    }

    // 将文件指针移回文件开头，准备读取
    file.seekg(0, std::ios::beg);

    // 创建一个vector来存储文件内容，大小为 (文件字节数 / 每个uint32_t的字节数)
    std::vector<uint32_t> buffer(static_cast<size_t>(fileSize) / sizeof(uint32_t));

    // 读取文件内容到buffer中
    // file.read() 需要一个 char* 类型的指针，所以我们用 reinterpret_cast
    if (file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        // YOUR_LOG_INFO("成功读取文件: %s, 大小: %lld字节",
        //               filename.c_str(), static_cast<long long>(fileSize));
        std::cout << "成功读取文件: " << filename << ", 大小: " << fileSize << "字节" << std::endl;
    } else {
        // YOUR_LOG_ERROR("错误: 从文件 '%s' 读取数据失败。", filename.c_str());
        std::cerr << "错误: 从文件 '" << filename << "' 读取数据失败。" << std::endl;
        file.close();
        return {}; // 读取失败，返回空vector
    }

    // 关闭文件 (虽然ifstream的析构函数会自动关闭，但显式关闭是个好习惯)
    file.close();

    return buffer;
}

// // 读取 SPV 文件
// std::vector<uint32_t> readSpvAsset(AAssetManager* mgr, const char* filename) {
//     AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_STREAMING);
//     if (!asset) {
//         std::cout << "Failed to open asset: " << filename << std::endl;
//         return {};
//     }
//     off_t length = AAsset_getLength(asset);
//     std::vector<uint32_t> buffer(length / sizeof(uint32_t));
//     AAsset_read(asset, buffer.data(), length);
//     AAsset_close(asset);
//     return buffer;
// }

} // namespace gpu_kompute