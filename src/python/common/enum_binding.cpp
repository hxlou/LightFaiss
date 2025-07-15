#include "src/python/common/enum_binding.hpp"

void bind_common_enums(py::module& m) {
    py::enum_<MetricType>(m, "MetricType")
    .value("METRIC_INNER_PRODUCT", MetricType::METRIC_INNER_PRODUCT)
    .value("METRIC_L2", MetricType::METRIC_L2)
    .export_values();
    
    // 绑定设备类型
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU_BLAS)
        .value("GPU", DeviceType::GPU_KOMPUTE)
        .value("NPU", DeviceType::NPU_HEXAGON)
        .export_values();
};