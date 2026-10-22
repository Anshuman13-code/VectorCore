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

DIM = 128
K = 5

# Generate unit-normalized sample data
data = np.random.randn(10000, DIM).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)
query = np.random.randn(DIM).astype(np.float32)
query /= np.linalg.norm(query)

# Load vectors into a dataset
dataset = vectorcore_py.FlatVectorDataset(DIM)
for vec in data:
    dataset.add_vector(vec.tolist())

# Build the HNSW index
index = vectorcore_py.HNSWIndex(dataset, M=32, ef_construction=128)
for i in range(len(data)):
    index.insert(i)

# Search for the nearest neighbors
index.set_ef_search(128)
scratch = vectorcore_py.SearchScratchpad(index.size())
results = index.query(query.tolist(), K=K, scratch=scratch)
print("Nearest neighbor IDs:", [node.id for node in results])
```

## ✅ Validating Recall

Recall is measured against an exact NumPy brute-force baseline. For 10,000 unit-normalized 128-dimensional vectors and 100 random queries, the audit computes the true top-5 neighbors by L2 distance and compares them with the HNSW results (`M=32`, `ef_construction=128`, `ef_search=128`). The reported **97.60% Recall@5** is the fraction of true neighbors the index returned.

## Repository Structure

* `include/` — Core C++20 headers and template definitions.
* `src/` — Engine implementation and pybind11 modules.
* `benchmarks/` — Scalability stress tests and SIMD profiling (`benchmark_sift1m`, `benchmark_metrics`).
* `tests/` — End-to-end binary persistence and memory round-trip checks.
