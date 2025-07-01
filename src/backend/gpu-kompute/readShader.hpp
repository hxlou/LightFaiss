#pragma once

#include <vector>
#include <string>
#include <fstream>  // 用于文件流操作
#include <iostream> // 用于错误输出 (你可以替换为你项目中的日志系统)
#include <cstdint>  // 用于 uint32_t

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

namespace gpu_kompute {

std::vector<uint32_t> readSpvFile(const std::string& filename);
std::vector<uint32_t> readSpvAsset(AAssetManager* mgr, const char* filename);

} // namespace gpu_kompute