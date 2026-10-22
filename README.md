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

## 💻 Quick Start & Usage

### 1. Build the C++ Engine & Python Bindings

```bash
mkdir build && cd build
cmake ..
make -j
```

### 2. Run in Python with Custom Data

You can import the compiled module and index your own high-dimensional data:

```python
import vectorcore_py
import numpy as np

# Generate sample 128-dimensional data
data = np.random.rand(10000, 128).astype(np.float32)
query = np.random.rand(1, 128).astype(np.float32)

# Initialize HNSW index and insert data
index = vectorcore_py.HNSWIndex(dim=128, max_elements=10000)
index.add_items(data)

# Perform sub-linear nearest neighbor search
labels, distances = index.search(query, k=10)
print("Nearest neighbor IDs:", labels)
```

## Repository Structure

* `include/` — Core C++20 headers and template definitions.
* `src/` — Engine implementation and pybind11 modules.
* `benchmarks/` — Scalability stress tests and SIMD profiling (`benchmark_sift1m`, `benchmark_metrics`).
* `tests/` — End-to-end binary persistence and memory round-trip checks.
