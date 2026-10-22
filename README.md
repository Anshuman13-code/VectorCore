# VectorCore

A high-performance, from-scratch vector search engine implemented in C++20, featuring a Hierarchical Navigable Small World (HNSW) graph index and custom Python bindings. Built with a focus on bare-metal system efficiency, cache locality, and SIMD hardware acceleration.

## 🚀 Results & Performance
* **Throughput & Scale:** Achieved **17,477 Queries Per Second (QPS)** on the 1-million-vector SIFT1M dataset (11-core multi-threaded).
* **Accuracy:** Maintained **99.6% Recall@10** at scale, and **97.60% Recall@5** during rigorous brute-force NumPy validation.
* **Hardware Acceleration:** Native ARM NEON SIMD intrinsics for L2 distance computation delivered a validated **3.90x bare-metal speedup** over scalar execution.

## Core Architecture
* **HNSW Graph Index:** Implements greedy routing and skip-list topology for sub-linear nearest neighbor search.
* **Zero-Copy Persistence:** Binary serialization of topological graphs and raw vector pools directly to disk with 100% bitwise consistency.
* **Python Interoperability:** Native pybind11 integration allowing seamless deployment in Python and exact-match recall validation via NumPy ground truth.

## Repository Structure
* `include/` — Core C++20 headers and template definitions.
* `src/` — Engine implementation and pybind11 modules.
* `benchmarks/` — Scalability stress tests and SIMD profiling (`benchmark_sift1m`, `benchmark_metrics`).
* `tests/` — End-to-end binary persistence and memory round-trip checks.
