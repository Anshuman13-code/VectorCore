import vectorcore_py
import random
import time

DIM = 128
NUM_VECTORS = 10000

print("1. Initializing C++ Dataset via Python...")
dataset = vectorcore_py.FlatVectorDataset(DIM)

# Generate some random vectors in Python
for i in range(NUM_VECTORS):
    vec = [random.uniform(0.0, 1.0) for _ in range(DIM)]
    dataset.add_vector(vec)

print(f"Dataset populated with {dataset.size()} vectors.")

print("2. Building HNSW Index...")
start = time.time()
index = vectorcore_py.HNSWIndex(dataset, M=16, ef_construction=64)
for i in range(NUM_VECTORS):
    index.insert(i)
print(f"Build complete in {time.time() - start:.3f} seconds.")

print("3. Querying Index...")
scratch = vectorcore_py.SearchScratchpad(index.size())
query_vec = [random.uniform(0.0, 1.0) for _ in range(DIM)]

index.set_ef_search(64)
results = index.query(query_vec, K=5, scratch=scratch)

for rank, res in enumerate(results):
    print(f"Rank {rank+1}: Node {res.id} (Distance: {res.dist:.4f})")