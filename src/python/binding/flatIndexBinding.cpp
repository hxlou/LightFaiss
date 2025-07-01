#include "src/python/binding/flatIndexBinding.hpp"
#include "src/python/wrapper/pyFlatIndex.hpp"
#include <pybind11/stl.h>

void bind_flat_index(py::module& m) {
    py::class_<PyFlatIndex>(m, "FlatIndex", R"pbdoc(
        Flat index for exact vector search.
        
        This index stores vectors in a flat structure and performs brute-force
        search for exact results. It supports both CPU and GPU computation.
    )pbdoc")
        .def(py::init<uint64_t, uint64_t, bool, MetricType>(),
             R"pbdoc(
                 Create a FlatIndex with specified parameters.
                 
                 Args:
                     dim: Vector dimension
                     capacity: Maximum number of vectors to store
                     isFloat16: Whether to use float16 storage (default: False)
                     metricType: Distance metric type (default: METRIC_INNER_PRODUCT)
             )pbdoc",
             py::arg("dim"), py::arg("capacity"), 
             py::arg("isFloat16") = false, 
             py::arg("metricType") = MetricType::METRIC_INNER_PRODUCT)
        
        .def(py::init<uint64_t>(),
             R"pbdoc(
                 Create a FlatIndex with dimension only.
                 
                 Args:
                     dim: Vector dimension
             )pbdoc",
             py::arg("dim"))
        
        .def("add_vectors", &PyFlatIndex::add_vectors,
             R"pbdoc(
                 Add vectors to the index.
                 
                 Args:
                     vectors: 2D numpy array of shape (n_vectors, dim)
             )pbdoc",
             py::arg("vectors"))
        
        .def("get_num", &PyFlatIndex::get_num,
             "Get the number of vectors in the index")
        
        .def("get_dim", &PyFlatIndex::get_dim,
             "Get the vector dimension")
        
        .def("get_capacity", &PyFlatIndex::get_capacity,
             "Get the index capacity")
        
        .def("is_float16", &PyFlatIndex::is_float16,
             "Check if using float16 storage")
        
        .def("query", &PyFlatIndex::query,
             R"pbdoc(
                 Query the index for nearest neighbors.
                 
                 Args:
                     queries: 2D numpy array of query vectors (n_queries, dim)
                     k: Number of nearest neighbors to return
                     device: Device type (CPU or GPU)
                 
                 Returns:
                     Tuple of (indices, distances) as numpy arrays
             )pbdoc",
             py::arg("queries"), py::arg("k"), py::arg("device"))
        
        .def("query_range", &PyFlatIndex::query_range,
             R"pbdoc(
                 Query the index within a specified range.
                 
                 Args:
                     queries: 2D numpy array of query vectors (n_queries, dim)
                     k: Number of nearest neighbors to return
                     start: Start index of the search range
                     end: End index of the search range
                     device: Device type (CPU or GPU)
                     transX: Transpose query matrix (default: False)
                     transY: Transpose database matrix (default: False)
                 
                 Returns:
                     Tuple of (indices, distances) as numpy arrays
             )pbdoc",
             py::arg("queries"), py::arg("k"), py::arg("start"), py::arg("end"), py::arg("device"))
        
        .def("reconstruct", &PyFlatIndex::reconstruct,
             R"pbdoc(
                 Reconstruct a vector by its index.
                 
                 Args:
                     idx: Vector index
                 
                 Returns:
                     Reconstructed vector as numpy array
             )pbdoc",
             py::arg("idx"))
        
        .def("save", &PyFlatIndex::save,
             R"pbdoc(
                 Save the index to a file.
                 
                 Args:
                     filename: Path to save the index
                 
                 Returns:
                     Status code (0 for success)
             )pbdoc",
             py::arg("filename"))
        
        .def("load", &PyFlatIndex::load,
             R"pbdoc(
                 Load the index from a file.
                 
                 Args:
                     filename: Path to load the index from
                 
                 Returns:
                     Status code (0 for success)
             )pbdoc",
             py::arg("filename"));
}