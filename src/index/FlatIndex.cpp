#include "index/FlatIndex.hpp"
#include "backend/cpu-blas/distance.hpp"
#include "backend/gpu-kompute/distance.hpp"


AAssetManager* FlatIndex::assetManager_ = nullptr;
std::mutex FlatIndex::assetManagerMutex_;

#include <filesystem>
#include <fstream>
#include <iostream>
#include "backend/npu-hexagon/calculator-api.h"
FlatIndex::FlatIndex(uint64_t dim, uint64_t capacity, bool isFloat16, MetricType metricType, kp::Manager* mgr)
        : dim_(dim), num_(0), capacity_(capacity), isFloat16_(isFloat16), metricType_(metricType), realMgr_() {
    // this->realMgr_ = kp::Manager();
    this->mgr_ = &this->realMgr_; // 使用实际的Kompute管理器实例
    data_.resize(capacity * dim);
    dataNorm_.resize(capacity * dim); // 初始化Norm数据
}

FlatIndex::FlatIndex(uint64_t dim, kp::Manager* mgr, MetricType metricType): realMgr_() {
    // 创建一个空的FlatIndex
    // this->realMgr_ = kp::Manager();
    this->mgr_ = &this->realMgr_; // 使用实际的Kompute管理器实例
    dim_ = dim;
    num_ = 0;
    capacity_ = 0;              // 默认容量
    isFloat16_ = false;         // 默认不使用 float16 存储
    metricType_ = metricType; // 默认使用内积度量
    mgr_ = mgr;                 // 默认不使用Kompute管理器
    data_.clear();              // 清空数据
    dataNorm_.clear();          // 清空Norm数据
}

void FlatIndex::addVector(const float* vecs, uint64_t n) {
    while ((num_ + n) >= capacity_)
        capacity_ = capacity_ == 0 ? 1 : capacity_ * 2; // 扩展容量，至少为1
    
    data_.resize(capacity_ * dim_);
    dataNorm_.resize(capacity_ * dim_);
    std::copy(vecs, vecs + n * dim_, data_.data() + num_ * dim_);
    
    // 预计算每个向量的Norm数据
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < dim_; ++j) {
            dataNorm_[num_ + i] += vecs[i * dim_ + j] * vecs[i * dim_ + j]; // 计算平方和
        }
    }
    
    num_ += n;
}

uint64_t FlatIndex::getNum() const {
    return num_;
}

uint64_t FlatIndex::getDim() const {
    return dim_;
}

uint64_t FlatIndex::getCapacity() const {
    return capacity_;
}

bool FlatIndex::isFloat16() const {
    return isFloat16_;
}

// 在指定设备上，对数据库的指定范围 [start,end) 进行查询
void FlatIndex::query(
    uint64_t k,
    uint64_t start,
    uint64_t end,
    DeviceType device,
    uint64_t nQuery,
    const float* query,
    uint64_t* results,
    float* distances
) {
    float* dataNorm = dataNorm_.empty() ? nullptr : dataNorm_.data();
	test_calculator();
    if (device == DeviceType::CPU_BLAS) {
        cpu_blas::query(
            nQuery,
            end - start,
            k,
            this->dim_,
            query,
            this->data_.data() + start * dim_,
            dataNorm + start * dim_,
            distances,
            results,
            metricType_
        );
    }
    else if (device == DeviceType::GPU_KOMPUTE) {
        gpu_kompute::query(
            mgr_,
            nQuery,
            end - start,
            k,
            this->dim_,
            query,
            this->data_.data() + start * dim_,
            dataNorm + start * dim_,
            distances,
            results,
            metricType_,
            nullptr
        );
    }
}

void FlatIndex::reconstruct(
    uint64_t idx,
    float* vec
) {
    if (idx >= num_) {
        // 索引超出范围
        return;
    }
    std::copy(data_.data() + idx * dim_, data_.data() + (idx + 1) * dim_, vec);
}

int FlatIndex::save(const std::string filename) {
    try {
        // 1. 将字符串转换为 filesystem::path 对象
        std::filesystem::path file_path(filename);

        // 2. 获取文件所在的目录 ("data/")
        std::filesystem::path dir_path = file_path.parent_path();

        // 3. 如果存在父目录，则检查并创建它
        //    (这可以避免对 "file1.bin" 这样的无目录路径执行操作)
        if (!dir_path.empty()) {
            // 4. 检查目录是否存在，如果不存在，则创建
            //    create_directories 类似于 `mkdir -p`，会创建所有必需的父目录
            //    如果目录已存在，它什么也不做。
            std::filesystem::create_directories(dir_path);
        }

    } catch (const std::filesystem::filesystem_error& e) {
        // 如果创建目录时发生 I/O 错误，捕获异常
        std::cerr << "Filesystem error while creating directory: " << e.what() << std::endl;
        return -1;
    }
    
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        return -1; // 打开文件失败
    }

    // header
    /*
        magicNumber + dim + num + isFloat16 + metricType
    */
    uint64_t magicNumber = 1145; 
    ofs.write(reinterpret_cast<const char*>(&magicNumber), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(&dim_), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(&num_), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(&isFloat16_), sizeof(bool));
    ofs.write(reinterpret_cast<const char*>(&metricType_), sizeof(MetricType));
    // true data
    ofs.write(reinterpret_cast<const char*>(data_.data()), num_ * dim_ * sizeof(float));
    ofs.write(reinterpret_cast<const char*>(dataNorm_.data()), num_ * dim_ * sizeof(float));

    ofs.close();
    return 0; // 成功
}

int FlatIndex::load(const std::string filename) {
    if (filename.find("data.") == std::string::npos) {
        std::string newFilename = "data/" + filename;
    }

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        return -1; // 打开文件失败
    }

    // 读取魔数
    uint64_t magicNumber;
    ifs.read(reinterpret_cast<char*>(&magicNumber), sizeof(uint64_t));
    if (magicNumber != 1145) {
        return -2; // 魔数不匹配
    }
    // 读取向量维度和数量
    ifs.read(reinterpret_cast<char*>(&dim_), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&num_), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&isFloat16_), sizeof(bool));
    ifs.read(reinterpret_cast<char*>(&metricType_), sizeof(MetricType));
    this->capacity_ = num_; // 设置容量为当前数量

    // 读取向量数据
    data_.resize(capacity_ * dim_);
    dataNorm_.resize(capacity_ * dim_);
    ifs.read(reinterpret_cast<char*>(data_.data()), num_ * dim_ * sizeof(float));
    ifs.read(reinterpret_cast<char*>(dataNorm_.data()), num_ * dim_ * sizeof(float));

    ifs.close();
    return 0; // 成功
}