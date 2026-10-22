#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

using namespace vectorcore;

// Helper to read .fvecs format
void read_fvecs(const std::string& filename, FlatVectorDataset& dataset, std::size_t expected_num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open " + filename);
    
    std::size_t dim = dataset.dimension();
    for (std::size_t i = 0; i < expected_num; ++i) {
        int d;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (d != dim) throw std::runtime_error("Dimension mismatch in fvecs");
        
        std::vector<float> vec(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        dataset.add_vector(vec);
    }
}

// Helper to read .ivecs format for ground truth
void read_ivecs(const std::string& filename, std::vector<std::vector<int>>& truth, std::size_t expected_num) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open " + filename);
    
    truth.resize(expected_num);
    for (std::size_t i = 0; i < expected_num; ++i) {
        int d; // number of ground truth neighbors (usually 100)
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        
        truth[i].resize(d);
        in.read(reinterpret_cast<char*>(truth[i].data()), d * sizeof(int));
    }
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "   VECTORCORE SIFT1M SCALE BENCHMARK\n";
    std::cout << "==========================================\n\n";

    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_BASE = 1000000;
    constexpr std::size_t NUM_QUERIES = 10000;
    constexpr std::size_t NUM_THREADS = 11; // M3 Pro Cores

    // 1. Load Data
    std::cout << "--- 1. Loading SIFT1M Dataset ---\n";
    FlatVectorDataset dataset(DIM);
    FlatVectorDataset queries(DIM);
    std::vector<std::vector<int>> ground_truth;

    read_fvecs("sift/sift_base.fvecs", dataset, NUM_BASE);
    std::cout << "Loaded " << dataset.size() << " base vectors.\n";
    
    read_fvecs("sift/sift_query.fvecs", queries, NUM_QUERIES);
    std::cout << "Loaded " << queries.size() << " query vectors.\n";
    
    read_ivecs("sift/sift_groundtruth.ivecs", ground_truth, NUM_QUERIES);

    // 2. Build Index (Heavy lifting)
    std::cout << "\n--- 2. Building HNSW Index (M=16, efConstruction=200) ---\n";
    std::cout << "Building 1 Million nodes... (This will take a few minutes)\n";
    auto build_start = std::chrono::high_resolution_clock::now();
    
    HNSWIndex index(dataset, 16, 200, 200, NUM_BASE);
    for (std::size_t i = 0; i < NUM_BASE; ++i) {
        index.insert(i);
        if ((i + 1) % 100000 == 0) std::cout << "Inserted " << (i + 1) << " / " << NUM_BASE << "\n";
    }
    
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "Build complete in " << build_sec << " seconds.\n";

    // 3. Multi-Threaded Query Evaluation
    std::cout << "\n--- 3. Multi-Core Evaluation (11 Threads) ---\n";
    std::vector<std::size_t> ef_sweeps = {10, 50, 100, 200};

    for (std::size_t ef : ef_sweeps) {
        index.set_ef_search(ef);
        
        std::atomic<std::size_t> recall_at_1{0};
        std::atomic<std::size_t> recall_at_10{0};
        
        auto eval_start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        std::atomic<std::size_t> current_query{0};

        for (std::size_t t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([&]() {
                SearchScratchpad scratch(index.size());
                while (true) {
                    std::size_t q_idx = current_query.fetch_add(1);
                    if (q_idx >= NUM_QUERIES) break;

                    auto results = index.query(queries.get_vector(q_idx), 10, scratch);
                    
                    // Evaluate Recall@1
                    if (!results.empty() && results[0].id == ground_truth[q_idx][0]) {
                        recall_at_1++;
                    }

                    // Evaluate Recall@10
                    std::size_t hits = 0;
                    for (const auto& res : results) {
                        for (std::size_t k = 0; k < 10; ++k) {
                            if (res.id == ground_truth[q_idx][k]) {
                                hits++;
                                break;
                            }
                        }
                    }
                    recall_at_10 += hits;
                }
            });
        }

        for (auto& thread : threads) thread.join();

        auto eval_end = std::chrono::high_resolution_clock::now();
        double eval_sec = std::chrono::duration<double>(eval_end - eval_start).count();
        double qps = NUM_QUERIES / eval_sec;

        double r1 = (double)recall_at_1 / NUM_QUERIES * 100.0;
        double r10 = (double)recall_at_10 / (NUM_QUERIES * 10) * 100.0;

        std::cout << "efSearch = " << std::setw(3) << ef 
                  << " | QPS: " << std::setw(6) << (int)qps 
                  << " | Recall@1: " << std::fixed << std::setprecision(1) << r1 << "%"
                  << " | Recall@10: " << r10 << "%\n";
    }

    return 0;
}