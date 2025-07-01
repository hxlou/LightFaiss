#pragma once
#include <pybind11/pybind11.h>

#include "src/index/MetricType.hpp"
#include "src/index/Device.hpp"

namespace py = pybind11;

void bind_common_enums(py::module& m);