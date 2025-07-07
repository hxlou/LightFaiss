#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "src/python/binding/flatIndexBinding.hpp"
#include "src/backend/cpu-blas/L2Norm.hpp"
#include "src/backend/gpu-kompute/L2Norm.hpp"

namespace py = pybind11;

void bind_L2_renorm(py::module& m) {

    m.def("normalized_L2_cpu",
        [](py::array_t<float> array, size_t dim, size_t nx) {
            py::buffer_info buf = array.request();
            float* ptr = static_cast<float*>(buf.ptr);
            cpu_blas::normalized_L2(dim, nx, ptr);
        },
        "CPU normalized L2 renorm",
        py::arg("array"), py::arg("dim"), py::arg("nx")
    );

    m.def("normalized_L2_gpu",
        [](py::object mgr_py_obj, py::array_t<float> array, size_t dim, size_t nx) {
            kp::Manager* mgr_ptr = nullptr;
            if (!mgr_py_obj.is_none()) {
                kp::Manager& mgr_ref = mgr_py_obj.cast<kp::Manager&>();
                mgr_ptr = &mgr_ref;
            }
            py::buffer_info buf = array.request();
            float* ptr = static_cast<float*>(buf.ptr);
            gpu_kompute::normalized_L2(mgr_ptr, dim, nx, ptr);
        },
        R"pbdoc(
            GPU normalized L2 renorm using Kompute.

            Args:
                mgr (kp.Manager): Kompute Manager instance.
                array (np.ndarray): Input array.
                dim (int): Vector dimension.
                nx (int): Number of vectors.
        )pbdoc",
        py::arg("mgr"), py::arg("array"), py::arg("dim"), py::arg("nx")
    );
}