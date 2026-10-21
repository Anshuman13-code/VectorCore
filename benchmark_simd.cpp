#include <iostream>
#include <vector>
#include <cstddef>
#include <random>
#include <chrono>
#include <limits>
#include <utility>
#include <arm_neon.h>

// -------------------------------------------------------------
// 1. Scalar L2 Distance
// -------------------------------------------------------------
float l2_scalar(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// -------------------------------------------------------------
// 2. ARM NEON SIMD L2 Distance
// -------------------------------------------------------------
float l2_neon(const float* a, const float* b, std::size_t dim) {
    float32x4_t acc = vdupq_n_f32(0.0f);

    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    float sum = vaddvq_f32(acc);

    // Handle leftover elements when dim is not a multiple of 4
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// -------------------------------------------------------------
// 3. Flat Vector Dataset
// -------------------------------------------------------------
class FlatVectorDataset {
private:
    std::size_t dimension_;
    std::vector<float> data_;

public:
    FlatVectorDataset(std::size_t dim) : dimension_(dim) {}

    void reserve(std::size_t num_vectors) {
        data_.reserve(num_vectors * dimension_);
    }

    void add_vector(const std::vector<float>& vec) {
        for (float val : vec) {
            data_.push_back(val);
        }
    }

    const float* get_vector(std::size_t index) const {
        return data_.data() + (index * dimension_);
    }

    std::size_t size() const {
        return data_.size() / dimension_;
    }

    std::size_t dimension() const {
        return dimension_;
    }
};

// -------------------------------------------------------------
// 4. Search Harnesses
// -------------------------------------------------------------
std::pair<std::size_t, float> search_scalar(const FlatVectorDataset& dataset, const float* query) {
    std::size_t best_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    const std::size_t total = dataset.size();
    const std::size_t dim = dataset.dimension();

    for (std::size_t i = 0; i < total; ++i) {
        float dist = l2_scalar(query, dataset.get_vector(i), dim);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return {best_idx, min_dist};
}

std::pair<std::size_t, float> search_neon(const FlatVectorDataset& dataset, const float* query) {
    std::size_t best_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    const std::size_t total = dataset.size();
    const std::size_t dim = dataset.dimension();

    for (std::size_t i = 0; i < total; ++i) {
        float dist = l2_neon(query, dataset.get_vector(i), dim);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return {best_idx, min_dist};
}

// -------------------------------------------------------------
// 5. Main Benchmark
// -------------------------------------------------------------
int main() {
    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_VECTORS = 10000;

    std::cout << "Generating dataset (" << NUM_VECTORS << " vectors, " << DIM << " dims)...\n";
    FlatVectorDataset dataset(DIM);
    dataset.reserve(NUM_VECTORS);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> temp(DIM);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        for (std::size_t d = 0; d < DIM; ++d) {
            temp[d] = dist(rng);
        }
        dataset.add_vector(temp);
    }

    const float* query = dataset.get_vector(5000);

    // Run Scalar Benchmark
    auto t1 = std::chrono::high_resolution_clock::now();
    auto [scalar_idx, scalar_dist] = search_scalar(dataset, query);
    auto t2 = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Run NEON Benchmark
    auto t3 = std::chrono::high_resolution_clock::now();
    auto [neon_idx, neon_dist] = search_neon(dataset, query);
    auto t4 = std::chrono::high_resolution_clock::now();
    double neon_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << "\n--- Benchmark Results ---\n";
    std::cout << "Scalar Search Time : " << scalar_ms << " ms | Nearest: " << scalar_idx
              << " (dist " << scalar_dist << ")\n";
    std::cout << "NEON SIMD Time     : " << neon_ms << " ms | Nearest: " << neon_idx
              << " (dist " << neon_dist << ")\n";
    std::cout << "Speedup Factor     : " << (scalar_ms / neon_ms) << "x\n";

    return 0;
}