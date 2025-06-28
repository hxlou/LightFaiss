#pragma once

#include <kompute/Kompute.hpp>

#include "MetricType.hpp"
#include "Device.hpp"

#include <vector>

class FlatIndex 
{
    public:
        FlatIndex(uint64_t dim, uint64_t capacity, bool isFloat16 = false, MetricType metricType = MetricType::METRIC_INNER_PRODUCT, kp::Manager* mgr = nullptr);
        ~FlatIndex() {};

        // 添加向量
        void addVector(const float* vecs, uint64_t n);
        // 获取向量数量
        uint64_t getNum() const;
        // 获取向量维度
        uint64_t getDim() const;
        // 获取向量容量
        uint64_t getCapacity() const;
        // 获取是否使用 float16 存储
        bool isFloat16() const;

        // 查询n个指定向量并返回前k个匹配的向量
        void query(
            uint64_t n,
            uint64_t k,
            DeviceType device,
            float* query,
            uint64_t* results,
            float* distances
        );

        // 在指定设备上，对数据库的指定范围 [start,end) 进行查询
        void query(
            uint64_t k,
            uint64_t start,
            uint64_t end,
            DeviceType device,
            uint64_t nQuery,
            const float* query,
            uint64_t* results,
            float* distances
        );

        // 根据索引重建向量
        void reconstruct(
            uint64_t idx,
            float* vec
        );

    private:
        kp::Manager* mgr_;             // Kompute管理器
        uint64_t dim_;                      // 向量维度
        uint64_t num_;                      // 向量数量    
        uint64_t capacity_;                 // 向量容量
        bool isFloat16_;                    // 是否使用 float16 存储
        MetricType metricType_;             // 距离计算方式
        std::vector<float> data_;           // 存储向量数据
        std::vector<float> dataNorm_;       // 存储向量归一化后的数据
};