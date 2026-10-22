#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <cmath>

using namespace vectorcore;

int main() {
    std::cout << "==========================================\n";
    std::cout << "   VECTORCORE PERSISTENCE VALIDATION\n";
    std::cout << "==========================================\n\n";

    constexpr std::size_t DIM = 64;
    constexpr std::size_t NUM_VECTORS = 5000;
    constexpr std::size_t NUM_TEST_QUERIES = 200;
    constexpr std::size_t K = 5;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::cout << "1. Building original dataset & index in-memory...\n";
    FlatVectorDataset original_dataset(DIM);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> v(DIM);
        for (std::size_t d = 0; d < DIM; ++d) v[d] = dist(rng);
        original_dataset.add_vector(v);
    }

    HNSWIndex original_index(original_dataset, 16, 64, 64, NUM_VECTORS);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        original_index.insert(i);
    }

    const std::string ds_path = "test_persistence_dataset.bin";
    const std::string idx_path = "test_persistence_index.bin";
    std::cout << "2. Serializing to " << ds_path << " and " << idx_path << "...\n";
    original_dataset.save(ds_path);
    original_index.save(idx_path);

    std::cout << "3. Deserializing from disk into fresh instances...\n";
    FlatVectorDataset loaded_dataset;
    loaded_dataset.load(ds_path);

    HNSWIndex loaded_index(loaded_dataset);
    loaded_index.load(idx_path);

    std::cout << "4. Verifying query consistency across " << NUM_TEST_QUERIES << " random queries...\n";
    SearchScratchpad scratch_orig(original_index.size());
    SearchScratchpad scratch_loaded(loaded_index.size());

    std::size_t mismatches = 0;
    constexpr float EPSILON = 1e-5f;

    for (std::size_t q = 0; q < NUM_TEST_QUERIES; ++q) {
        std::vector<float> query_vec(DIM);
        for (std::size_t d = 0; d < DIM; ++d) query_vec[d] = dist(rng);

        auto res_orig = original_index.query(query_vec.data(), K, scratch_orig);
        auto res_loaded = loaded_index.query(query_vec.data(), K, scratch_loaded);

        if (res_orig.size() != res_loaded.size()) {
            mismatches++;
            continue;
        }

        for (std::size_t i = 0; i < res_orig.size(); ++i) {
            if (res_orig[i].id != res_loaded[i].id || std::abs(res_orig[i].dist - res_loaded[i].dist) > EPSILON) {
                mismatches++;
                break;
            }
        }
    }

    std::cout << "------------------------------------------\n";
    std::cout << "Dataset Dimensions Loaded : " << loaded_dataset.dimension() << " (Expected: " << DIM << ")\n";
    std::cout << "Dataset Vector Count Loaded: " << loaded_dataset.size() << " (Expected: " << NUM_VECTORS << ")\n";
    std::cout << "Index Node Count Loaded   : " << loaded_index.size() << " (Expected: " << NUM_VECTORS << ")\n";
    std::cout << "Validation Mismatches     : " << mismatches << "\n";
    std::cout << "------------------------------------------\n";

    if (mismatches == 0) {
        std::cout << "STATUS: PASS (100% Bitwise/Topological Consistency)\n\n";
    } else {
        std::cerr << "STATUS: FAIL (Detected state corruption during round-trip)\n\n";
    }

    std::filesystem::remove(ds_path);
    std::filesystem::remove(idx_path);

    return (mismatches == 0) ? 0 : 1;
}
