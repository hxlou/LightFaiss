#include "index/FlatIndex.hpp"
#include "backend/cpu-blas/distance.hpp"
#include "backend/gpu-kompute/distance.hpp"


AAssetManager* FlatIndex::assetManager_ = nullptr;
std::mutex FlatIndex::assetManagerMutex_;

#include <vector>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
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

void FlatIndex::search(
    uint64_t k,
    uint64_t nQuery,
    const float* query,
    uint64_t* results,
    float* distances
) {
    // 在这一层进行调度
    // 如果nQ * nY > 10000，则使用GPU，否则使用CPU
    uint64_t nData = num_;
    uint64_t threshold = 10000;

    if (nQuery * nData <= threshold) {
        // 计算量小的情况下，直接调用CPU完成计算并返回结果
        this->query(
            k,
            0,
            nData,
            DeviceType::CPU_BLAS,
            nQuery,
            query,
            results,
            distances
        );

        return;
    }

    /**
     * 进行调度计算，保证每个后端的data的数量均满足>=k，这样为每个计算后端
     * 预先分配的中间结果临时空间会直接填满，不会有空闲空间，方便后续汇总
     * 处理
     */
    std::vector<uint64_t> resultsTmpCPU(nQuery * k, 0);
    std::vector<uint64_t> resultsTmpGPU(nQuery * k, 0);
    std::vector<uint64_t> resultsTmpNPU(nQuery * k, 0);
    
    std::vector<float> distancesTmpCPU(nQuery * k, 0.0f);
    std::vector<float> distancesTmpGPU(nQuery * k, 0.0f);
    std::vector<float> distancesTmpNPU(nQuery * k, 0.0f);
    
    // 任务分配（TODO 任务关键，需要结合硬件负载来进行）
    uint64_t cpu_start  = 0;
    uint64_t cpu_end    = num_ / 2;            // [start_cpu, end_cpu)
    uint64_t gpu_start  = num_ / 2;
    uint64_t gpu_end    = num_;                // [start_gpu, end_gpu)
    uint64_t npu_start  = UINT64_MAX;
    uint64_t npu_end    = UINT64_MAX;          // [start_npu, end_npu)

    std::thread cpu_thread;
    std::thread gpu_thread;
    std::thread npu_thread;

    // 现有的默认分配：CPU和GPU各分配一半，NPU不分配
    if (cpu_start != UINT64_MAX) {
        cpu_thread = std::thread(
            [&](){
                this->query(k, cpu_start, cpu_end, DeviceType::CPU_BLAS, nQuery, query, resultsTmpCPU.data(), distancesTmpCPU.data());
            }
        );
    }

    if (gpu_start != UINT64_MAX) {
        gpu_thread = std::thread(
            [&](){
                this->query(k, gpu_start, gpu_end, DeviceType::GPU_KOMPUTE, nQuery, query, resultsTmpGPU.data(), distancesTmpGPU.data());
            }
        );
    }

    if (npu_start != UINT64_MAX) {
        npu_thread = std::thread(
            [&](){
                this->query(k, npu_start, npu_end, DeviceType::NPU_HEXAGON, nQuery, query, resultsTmpNPU.data(), distancesTmpNPU.data());
            }
        );
    }
    
    if (cpu_start != UINT64_MAX) {
        cpu_thread.join();
    }
    if (gpu_start != UINT64_MAX) {
        gpu_thread.join();
    }
    if (npu_start != UINT64_MAX) {
        npu_thread.join();
    }

    // 汇总结果
    // IP，最终结果降序（内积结果越大越好）
    bool isDesc = (this->metricType_ == MetricType::METRIC_INNER_PRODUCT) ? true : false;
    uint64_t backend_nums = ((cpu_start != UINT64_MAX ? 1 : 0) +
                             (gpu_start != UINT64_MAX ? 1 : 0) +
                             (npu_start != UINT64_MAX ? 1 : 0));

    #pragma omp parallel for
    for (uint64_t i = 0; i < nQuery; ++i) {
        std::vector<std::pair<float, uint64_t>> results_tmp(k * backend_nums);
        // CPU结果
        if (cpu_start != UINT64_MAX) {
            for (uint64_t j = 0; j < k; ++j) {
                float value = distancesTmpCPU[i * k + j];
                results_tmp[j] = std::make_pair(value, resultsTmpCPU[i * k + j]);
            }
        }
        // GPU结果
        if (gpu_start != UINT64_MAX) {
            for (uint64_t j = 0; j < k; ++j) {
                float value = distancesTmpGPU[i * k + j];
                results_tmp[j + k] = std::make_pair(value, resultsTmpGPU[i * k + j]);
            }
        }
        // NPU结果
        if (npu_start != UINT64_MAX) {
            for (uint64_t j = 0; j < k; ++j) {
                float value = distancesTmpNPU[i * k + j];
                results_tmp[j + 2 * k] = std::make_pair(value, resultsTmpNPU[i * k + j]);
            }
        }

        // 排序并取前k个结果
        std::sort(results_tmp.begin(), results_tmp.end(),
                          [&](const std::pair<float, uint64_t>& a, const std::pair<float, uint64_t>& b) {
                                if (isDesc) {
                                    return a.first > b.first; // 降序排序
                                }
                                return a.first < b.first; // 升序排序
                          });

        // 将前k个结果写入输出
        for (uint64_t j = 0; j < k; ++j) {
            distances[i * k + j] = results_tmp[j].first;
            results[i * k + j] = results_tmp[j].second;
        }
    }

    return;
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