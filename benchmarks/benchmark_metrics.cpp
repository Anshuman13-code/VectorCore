#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>

using namespace vectorcore;

// Pure scalar implementation to compare against our engine's SIMD l2_distance
float scalar_l2_distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "   VECTORCORE METRICS & PROFILING\n";
    std::cout << "==========================================\n\n";

    // --- 1. SIMD VS SCALAR MICRO-BENCHMARK ---
    std::cout << "--- 1. ARM NEON SIMD vs Scalar Fallback ---\n";
    constexpr std::size_t DIM = 128;
    constexpr std::size_t ITERS = 10000000; // 10 Million distance calculations
    
    std::vector<float> vec_a(DIM), vec_b(DIM);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(std::size_t i = 0; i < DIM; ++i) {
        vec_a[i] = dist(rng);
        vec_b[i] = dist(rng);
    }

    // Benchmark Scalar
    auto t1 = std::chrono::high_resolution_clock::now();
    float dummy_scalar = 0;
    for(std::size_t i = 0; i < ITERS; ++i) dummy_scalar += scalar_l2_distance(vec_a.data(), vec_b.data(), DIM);
    auto t2 = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Benchmark SIMD (using engine's native l2_distance)
    auto t3 = std::chrono::high_resolution_clock::now();
    float dummy_simd = 0;
    for(std::size_t i = 0; i < ITERS; ++i) dummy_simd += l2_distance(vec_a.data(), vec_b.data(), DIM);
    auto t4 = std::chrono::high_resolution_clock::now();
    double simd_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // Prevent aggressive compiler optimization from stripping the loops
    if (dummy_scalar == -1.0f || dummy_simd == -1.0f) std::cout << "Ignore"; 

    std::cout << "Distance ops   : " << ITERS << " calls (Dim: " << DIM << ")\n";
    std::cout << "Scalar Time    : " << scalar_ms << " ms\n";
    std::cout << "SIMD Time      : " << simd_ms << " ms\n";
    std::cout << "SIMD Speedup   : " << std::fixed << std::setprecision(2) << (scalar_ms / simd_ms) << "x faster\n\n";


    // --- 2. MEMORY OVERHEAD PROFILING ---
    std::cout << "--- 2. HNSW Graph Memory Overhead ---\n";
    constexpr std::size_t NUM_VECTORS = 10000;
    
    FlatVectorDataset dataset(DIM);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> v(DIM);
        for(std::size_t d = 0; d < DIM; ++d) v[d] = dist(rng);
        dataset.add_vector(v);
    }

    // Standard M=16 production parameters
    HNSWIndex index(dataset, 16, 64, 64, NUM_VECTORS);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        index.insert(i);
    }

    dataset.save("metrics_dataset.bin");
    index.save("metrics_index.bin");

    auto ds_size = std::filesystem::file_size("metrics_dataset.bin");
    auto idx_size = std::filesystem::file_size("metrics_index.bin");
    double ds_mb = ds_size / (1024.0 * 1024.0);
    double idx_mb = idx_size / (1024.0 * 1024.0);

    std::cout << "Vectors        : " << NUM_VECTORS << " (Dim: " << DIM << ")\n";
    std::cout << "Dataset Size   : " << ds_mb << " MB (Raw Floats)\n";
    std::cout << "Index Size     : " << idx_mb << " MB (Graph Topology)\n";
    std::cout << "Graph Tax      : " << (idx_mb / ds_mb) * 100.0 << "% overhead\n";
    
    // Cleanup temporary metric files
    std::filesystem::remove("metrics_dataset.bin");
    std::filesystem::remove("metrics_index.bin");

    return 0;
}