#include "vectorcore/hnsw.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace vectorcore;

// Pure scalar implementation for baseline
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

    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_VECTORS = 10000;
    constexpr std::size_t ITERS = 10000000; 

    // 1. Generate a pool of distinct vectors to prevent compiler hoisting
    FlatVectorDataset dataset(DIM);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> v(DIM);
        for(std::size_t d = 0; d < DIM; ++d) v[d] = dist(rng);
        dataset.add_vector(v);
    }

    std::cout << "--- 1. ARM NEON SIMD vs Scalar Fallback ---\n";
    
    // Benchmark Scalar
    auto t1 = std::chrono::high_resolution_clock::now();
    float scalar_sink = 0.0f;
    for(std::size_t i = 0; i < ITERS; ++i) {
        const float* a = dataset.get_vector(i % NUM_VECTORS);
        const float* b = dataset.get_vector((i * 7 + 1) % NUM_VECTORS);
        scalar_sink += scalar_l2_distance(a, b, DIM);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Benchmark SIMD
    auto t3 = std::chrono::high_resolution_clock::now();
    float simd_sink = 0.0f;
    for(std::size_t i = 0; i < ITERS; ++i) {
        const float* a = dataset.get_vector(i % NUM_VECTORS);
        const float* b = dataset.get_vector((i * 7 + 1) % NUM_VECTORS);
        simd_sink += l2_distance(a, b, DIM);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double simd_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << "Distance ops   : " << ITERS << " calls (Dim: " << DIM << ")\n";
    std::cout << "Scalar Time    : " << scalar_ms << " ms\n";
    std::cout << "SIMD Time      : " << simd_ms << " ms\n";
    std::cout << "SIMD Speedup   : " << std::fixed << std::setprecision(2) << (scalar_ms / simd_ms) << "x faster\n";
    
    // Print checksums to prevent dead-code elimination and verify mathematical accuracy
    std::cout << "Scalar Checksum: " << scalar_sink << "\n";
    std::cout << "SIMD Checksum  : " << simd_sink << "\n\n";

    return 0;
}