#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>

using namespace vectorcore;

// Parser for SIFT .fvecs files (vectors)
bool load_fvecs(const std::string& filename, FlatVectorDataset& dataset) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << filename << "\n";
        return false;
    }

    uint32_t dim;
    std::vector<float> vec;
    
    while (in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        vec.resize(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        dataset.add_vector(vec);
    }
    return true;
}

// Parser for SIFT .ivecs files (ground truth)
bool load_ivecs(const std::string& filename, std::vector<std::vector<uint32_t>>& ground_truth) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << filename << "\n";
        return false;
    }

    uint32_t num_neighbors;
    std::vector<uint32_t> neighbors;
    
    while (in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors))) {
        neighbors.resize(num_neighbors);
        in.read(reinterpret_cast<char*>(neighbors.data()), num_neighbors * sizeof(uint32_t));
        ground_truth.push_back(neighbors);
    }
    return true;
}

int main() {
    std::cout << "--- 1. Loading Real SIFT Data ---\n";
    FlatVectorDataset train_data;
    FlatVectorDataset test_data;
    std::vector<std::vector<uint32_t>> ground_truth;

    if (!load_fvecs("sift/siftsmall_base.fvecs", train_data) ||
        !load_fvecs("sift/siftsmall_query.fvecs", test_data) ||
        !load_ivecs("sift/siftsmall_groundtruth.ivecs", ground_truth)) {
        std::cerr << "Error: Could not load SIFT files. Did you download and extract them?\n";
        return 1;
    }

    std::cout << "Loaded " << train_data.size() << " base vectors (dim=" << train_data.dimension() << ")\n";
    std::cout << "Loaded " << test_data.size() << " query vectors\n";

    std::cout << "\n--- 2. Building HNSW Index ---\n";
    auto t_build_start = std::chrono::high_resolution_clock::now();
    
    // Using standard production parameters (M=16, efConstruction=200 for real data)
    HNSWIndex index(train_data, 16, 200, 64, train_data.size());
    for (std::size_t i = 0; i < train_data.size(); ++i) {
        index.insert(static_cast<NodeId>(i));
    }
    
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << "Index built in: " << build_sec << " s | Max Level: " << index.max_level() << "\n\n";

    std::cout << "--- 3. Evaluating on SIFT Ground Truth ---\n";
    std::vector<std::size_t> ef_sweep = {16, 32, 64, 128, 256};
    
    // We only need top-10 recall for standard benchmarking
    constexpr std::size_t K = 10;
    const std::size_t NUM_TEST = test_data.size();
    
    SearchScratchpad scratch(index.size());
    std::vector<std::vector<DistNode>> query_results(NUM_TEST);

    for (std::size_t ef : ef_sweep) {
        index.set_ef_search(ef);

        // Time the search phase strictly
        auto t_start = std::chrono::high_resolution_clock::now();
        for (std::size_t q = 0; q < NUM_TEST; ++q) {
            query_results[q] = index.query(test_data.get_vector(q), K, scratch);
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double mean_latency_ms = total_ms / NUM_TEST;
        double qps = NUM_TEST / (total_ms / 1000.0);

        // Verify Recall@10 against the loaded .ivecs ground truth
        std::size_t total_matches = 0;
        for (std::size_t q = 0; q < NUM_TEST; ++q) {
            for (const auto& res : query_results[q]) {
                for (std::size_t gt_idx = 0; gt_idx < K; ++gt_idx) {
                    if (res.id == ground_truth[q][gt_idx]) {
                        total_matches++;
                        break;
                    }
                }
            }
        }
        double recall = (static_cast<double>(total_matches) / (NUM_TEST * K)) * 100.0;

        std::cout << "efSearch = " << ef
                  << "\t| Latency: " << mean_latency_ms << " ms"
                  << "\t| QPS: " << static_cast<int>(qps)
                  << "\t| Recall@" << K << ": " << recall << "%\n";
    }

    return 0;
}