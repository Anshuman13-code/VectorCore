# VectorCore

A high-performance, from-scratch vector search engine implemented in C++20, featuring a Hierarchical Navigable Small World (HNSW) graph index and custom Python bindings. 

Built with a focus on bare-metal system efficiency, cache locality, and SIMD hardware acceleration.

## Core Architecture
* **HNSW Graph Index:** Implements greedy routing and skip-list topology for sub-linear nearest neighbor search.
* **Hardware Acceleration:** Native ARM NEON SIMD intrinsics for L2 distance computation, achieving a validated **3.90x bare-metal speedup** over scalar execution.
* **Zero-Copy Persistence:** Binary serialization of topological graphs and raw vector pools directly to disk with 100% bitwise consistency.
* **Python Interoperability:** Native pybind11 integration allowing seamless deployment in Python and exact-match recall validation via NumPy ground truth.

## Scale Benchmark (SIFT1M)
Evaluated on the 1-million-vector SIFT1M dataset across an 11-core multi-threaded environment:
* **Throughput:** 17,477 Queries Per Second (QPS)
* **Accuracy:** 99.6% Recall@10

## Python NumPy Validation
Python bindings undergo a rigorous brute-force audit, consistently achieving **97.60% Recall@5** on normalized vector distributions compared to exact mathematical L2 baselines.

## Repository Structure
* `include/` — Core C++20 headers and template definitions.
* `src/` — Engine implementation and pybind11 modules.
* `benchmarks/` — Scalability stress tests and SIMD profiling (`benchmark_sift1m`, `benchmark_metrics`).
* `tests/` — End-to-end binary persistence and memory round-trip checks.
