#include <pybind11/pybind11.h>

#include "src/python/binding/flatIndexBinding.hpp"
#include "src/python/binding/L2RenormBinding.hpp"
#include "src/python/common/enum_binding.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lightfaiss_py, m) {
    m.doc() = "LightFaiss Python bindings - A lightweight vector search library";
    
    py::module_::import("kp");
    // 绑定通用枚举和类型
    bind_common_enums(m);
    
    // 绑定各种索引
    bind_flat_index(m);
    // bind_hnsw_index(m);  // 当你有其他索引时添加
    // bind_ivf_index(m);   // 当你有其他索引时添加

    // 绑定L2归一化
    bind_L2_renorm(m);
}