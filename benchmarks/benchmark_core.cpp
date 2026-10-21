#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <future>

using namespace vectorcore;

// ... (Keep the brute_force_knn and generate_clustered_data functions exactly as they were before) ...
std::vector<DistNode> brute_force_knn(const FlatVectorDataset& dataset, const float* query, std::size_t K) {
    std::priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> max_heap;
    const std::size_t total = dataset.size();
    const std::size_t dim = dataset.dimension();
    for (std::size_t i = 0; i < total; ++i) {
        float d = l2_distance(query, dataset.get_vector(i), dim);
        if (max_heap.size() < K) max_heap.push({d, static_cast<NodeId>(i)});
        else if (d < max_heap.top().dist) { max_heap.pop(); max_heap.push({d, static_cast<NodeId>(i)}); }
    }
    std::vector<DistNode> res;
    res.reserve(max_heap.size());
    while (!max_heap.empty()) { res.push_back(max_heap.top()); max_heap.pop(); }
    std::reverse(res.begin(), res.end());
    return res;
}

void generate_clustered_data(FlatVectorDataset& train, FlatVectorDataset& test, std::size_t num_train, std::size_t num_test, std::size_t dim, std::size_t num_clusters = 8) {
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> cluster_dist(0, num_clusters - 1);
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim));
    for (std::size_t c = 0; c < num_clusters; ++c) {
        for (std::size_t d = 0; d < dim; ++d) centroids[c][d] = norm_dist(rng) * 5.0f;
    }
    auto fill_dataset = [&](FlatVectorDataset& ds, std::size_t count) {
        std::vector<float> vec(dim);
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t c = cluster_dist(rng);
            for (std::size_t d = 0; d < dim; ++d) vec[d] = centroids[c][d] + norm_dist(rng) * 0.5f;
            ds.add_vector(vec);
        }
    };
    fill_dataset(train, num_train);
    fill_dataset(test, num_test);
}

int main() {
    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_TRAIN = 10000;
    constexpr std::size_t NUM_TEST = 200; // Used for recall accuracy testing
    constexpr std::size_t K = 10;

    std::cout << "--- 1. Generating Clustered GMM Dataset ---\n";
    FlatVectorDataset train_data(DIM);
    FlatVectorDataset test_data(DIM);
    train_data.reserve(NUM_TRAIN);
    test_data.reserve(NUM_TEST);
    generate_clustered_data(train_data, test_data, NUM_TRAIN, NUM_TEST, DIM, 8);

    std::cout << "--- 2. Building Production HNSW Index ---\n";
    HNSWIndex index(train_data, 16, 64, 64, NUM_TRAIN);
    for (std::size_t i = 0; i < NUM_TRAIN; ++i) {
        index.insert(static_cast<NodeId>(i));
    }
    
    // We now instantiate a SearchScratchpad for our queries on the main thread
    SearchScratchpad main_scratch(index.size());
    index.set_ef_search(64);

    std::cout << "--- 3. Single-Threaded vs. Multi-Threaded Throughput Benchmark ---\n";
    
    // To measure pure throughput, we will fire 10,000 queries.
    constexpr std::size_t BENCHMARK_QUERIES = 10000;
    std::vector<const float*> stress_queries(BENCHMARK_QUERIES);
    for (std::size_t i = 0; i < BENCHMARK_QUERIES; ++i) {
        stress_queries[i] = test_data.get_vector(i % NUM_TEST); // reuse test data to hit 10,000
    }

    // Single-Threaded Run
    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < BENCHMARK_QUERIES; ++i) {
        volatile auto sink = index.query(stress_queries[i], K, main_scratch);
        (void)sink;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double st_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double st_qps = BENCHMARK_QUERIES / (st_ms / 1000.0);

    // Multi-Threaded Run
    auto t3 = std::chrono::high_resolution_clock::now();
    
    std::size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::size_t chunk_size = BENCHMARK_QUERIES / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Each thread provisions its own lock-free scratchpad
            SearchScratchpad thread_local_scratch(index.size());
            
            std::size_t start = t * chunk_size;
            std::size_t end = (t == num_threads - 1) ? BENCHMARK_QUERIES : start + chunk_size;
            
            for (std::size_t i = start; i < end; ++i) {
                volatile auto sink = index.query(stress_queries[i], K, thread_local_scratch);
                (void)sink;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double mt_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double mt_qps = BENCHMARK_QUERIES / (mt_ms / 1000.0);

    std::cout << "Hardware Threads  : " << num_threads << "\n";
    std::cout << "Single-Core QPS   : " << static_cast<int>(st_qps) << " queries/sec\n";
    std::cout << "Multi-Core QPS    : " << static_cast<int>(mt_qps) << " queries/sec\n";
    std::cout << "Scalability       : " << (mt_qps / st_qps) << "x speedup\n";

    return 0;
}