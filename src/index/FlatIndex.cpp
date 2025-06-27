#include "index/FlatIndex.hpp"
#include "backend/cpu-blas/distance.hpp"
#include "backend/gpu-kompute/distance.hpp"



FlatIndex::FlatIndex(uint64_t dim, uint64_t capacity, bool isFloat16, MetricType metricType, kp::Manager* mgr)
    : dim_(dim), num_(0), capacity_(capacity), isFloat16_(isFloat16), metricType_(metricType), mgr_(mgr) {
    data_.resize(capacity * dim);
    dataNorm_.resize(capacity * dim); // 初始化Norm数据
}

void FlatIndex::addVector(const float* vecs, uint64_t n) {
    while ((num_ + n) >= capacity_)
        capacity_ *= 2;
    data_.resize(capacity_ * dim_);
    dataNorm_.resize(capacity_ * dim_);
    std::copy(vecs, vecs + n * dim_, data_.data() + num_ * dim_);
    
    // 预计算每个向量的Norm数据
    for (uint64_t i = 0; i < n * dim_; ++i) {
        dataNorm_[num_ * dim_ + i] = data_[num_ * dim_ + i] * data_[num_ * dim_ + i];
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
            this->num_,
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
            this->num_,
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

}
