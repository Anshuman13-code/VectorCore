import vectorcore_py
import numpy as np
import time

DIM = 128
NUM_VECTORS = 10000
NUM_QUERIES = 100
K = 5

print("==========================================")
print("   PYTHON INTEROP & NUMPY RECALL AUDIT   ")
print("==========================================\n")

np.random.seed(42)
# Generate unit-normalized embeddings (standard for vector databases)
raw_data = np.random.randn(NUM_VECTORS, DIM).astype(np.float32)
raw_data /= np.linalg.norm(raw_data, axis=1, keepdims=True)

raw_queries = np.random.randn(NUM_QUERIES, DIM).astype(np.float32)
raw_queries /= np.linalg.norm(raw_queries, axis=1, keepdims=True)

# 1. Ingest Data
print(f"1. Ingesting {NUM_VECTORS} vectors into FlatVectorDataset...")
dataset = vectorcore_py.FlatVectorDataset(DIM)
for i in range(NUM_VECTORS):
    dataset.add_vector(raw_data[i].tolist())

# 2. Build Index with higher connectivity for random distributions
print("2. Building HNSW Index (M=32, efConstruction=128)...")
t0 = time.time()
index = vectorcore_py.HNSWIndex(dataset, M=32, ef_construction=128)
for i in range(NUM_VECTORS):
    index.insert(i)
print(f"Index built in {time.time() - t0:.3f} seconds.")

# 3. Query with expanded search beam
print(f"\n3. Running {NUM_QUERIES} queries against NumPy brute-force...")
index.set_ef_search(128)
scratch = vectorcore_py.SearchScratchpad(index.size())

total_hits = 0

for q_idx in range(NUM_QUERIES):
    q_vec = raw_queries[q_idx]

    # NumPy Brute-Force L2 Ground Truth
    diffs = raw_data - q_vec
    sq_dists = np.sum(diffs ** 2, axis=1)
    exact_top_k = set(np.argsort(sq_dists)[:K])

    # VectorCore Query
    hnsw_results = index.query(q_vec.tolist(), K=K, scratch=scratch)
    hnsw_top_k = set(node.id for node in hnsw_results)

    hits = len(exact_top_k.intersection(hnsw_top_k))
    total_hits += hits

recall = (total_hits / (NUM_QUERIES * K)) * 100.0

print("------------------------------------------")
print(f"Total Evaluated Queries : {NUM_QUERIES}")
print(f"Top-K per Query         : {K}")
print(f"Empirical Recall@{K}    : {recall:.2f}%")
print("------------------------------------------")

if recall >= 95.0:
    print("STATUS: PASS (High-recall convergence verified via Python)\n")
else:
    print("STATUS: REVIEW\n")