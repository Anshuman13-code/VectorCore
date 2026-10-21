#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector <-> Python lists
#include "vectorcore/hnsw.hpp"
#include <string>

namespace py = pybind11;
using namespace vectorcore;

PYBIND11_MODULE(vectorcore_py, m) {
    m.doc() = "VectorCore Python Bindings for HNSW Index";

    // Expose DistNode so Python can read the results
    py::class_<DistNode>(m, "DistNode")
        .def_readonly("id", &DistNode::id)
        .def_readonly("dist", &DistNode::dist)
        .def("__repr__", [](const DistNode& n) {
            return "<DistNode id=" + std::to_string(n.id) + " dist=" + std::to_string(n.dist) + ">";
        });

    // Expose the Dataset
    py::class_<FlatVectorDataset>(m, "FlatVectorDataset")
        .def(py::init<std::size_t>(), py::arg("dim") = 0)
        .def("add_vector", &FlatVectorDataset::add_vector)
        .def("size", &FlatVectorDataset::size)
        .def("dimension", &FlatVectorDataset::dimension)
        .def("save", &FlatVectorDataset::save)
        .def("load", &FlatVectorDataset::load);

    // Expose the Thread-Local Scratchpad
    py::class_<SearchScratchpad>(m, "SearchScratchpad")
        .def(py::init<std::size_t>())
        .def("resize", &SearchScratchpad::resize);

    // Expose the Engine
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<const FlatVectorDataset&, std::size_t, std::size_t, std::size_t, std::size_t>(),
             py::arg("dataset"), py::arg("M") = 16, py::arg("ef_construction") = 64, 
             py::arg("ef_search") = 64, py::arg("expected_size") = 10000,
             // Ties the lifecycle of the dataset (arg 2) to the index (arg 1)
             // preventing Python's Garbage Collector from causing a segfault.
             py::keep_alive<1, 2>()) 
        .def("insert", &HNSWIndex::insert)
        .def("set_ef_search", &HNSWIndex::set_ef_search)
        // Lambda wrapper to convert Python list -> std::vector -> const float*
        .def("query", [](const HNSWIndex& self, const std::vector<float>& query_vec, std::size_t K, SearchScratchpad& scratch) {
            return self.query(query_vec.data(), K, scratch);
        }, py::arg("query_vec"), py::arg("K"), py::arg("scratch"))
        .def("save", &HNSWIndex::save)
        .def("load", &HNSWIndex::load)
        .def("size", &HNSWIndex::size);
}