#include <iostream>
#include <vector>
#include <cstddef>
#include <random>
#include <chrono>
#include <limits>

// -------------------------------------------------------------
// 1. The Distance Function (from our previous step)
// -------------------------------------------------------------
float l2_squared(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// -------------------------------------------------------------
// 2. The Flat Memory Dataset (from our first step)
// -------------------------------------------------------------
class FlatVectorDataset {
private:
    std::size_t dimension_;
    std::vector<float> data_;

public:
    FlatVectorDataset(std::size_t dim) : dimension_(dim) {}

    // Pre-allocate memory so vector doesn't keep reallocating as it grows
    void reserve(std::size_t num_vectors) {
        data_.reserve(num_vectors * dimension_);
    }

    void add_vector(const std::vector<float>& vec) {
        for (float val : vec) {
            data_.push_back(val);
        }
    }

    // Direct pointer to where vector 'index' starts
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
// 3. The Brute-Force Linear Search
// -------------------------------------------------------------
// Loops through every vector in the dataset, computes distance to query,
// and returns: pair<best_index, smallest_distance>
std::pair<std::size_t, float> brute_force_search(
    const FlatVectorDataset& dataset,
    const float* query
) {
    std::size_t best_index = 0;
    float min_dist = std::numeric_limits<float>::max(); // Start with infinity
    const std::size_t total = dataset.size();
    const std::size_t dim = dataset.dimension();

    for (std::size_t i = 0; i < total; ++i) {
        // Grab pointer to vector i
        const float* candidate = dataset.get_vector(i);

        // Compute geometric distance
        float dist = l2_squared(query, candidate, dim);

        // Check if this is the closest vector seen so far
        if (dist < min_dist) {
            min_dist = dist;
            best_index = i;
        }
    }

    return {best_index, min_dist};
}

// -------------------------------------------------------------
// 4. Main Test Harness
// -------------------------------------------------------------
int main() {
    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_VECTORS = 10000;

    std::cout << "Allocating and generating " << NUM_VECTORS 
              << " vectors of dimension " << DIM << "...\n";

    FlatVectorDataset dataset(DIM);
    dataset.reserve(NUM_VECTORS);

    // Fill with random floats between 0.0 and 1.0
    // std::mt19937 is a standard pseudo-random number generator
    std::mt19937 rng(42); // Seed = 42 for repeatable results
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> temp_vec(DIM);
    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        for (std::size_t d = 0; d < DIM; ++d) {
            temp_vec[d] = dist(rng);
        }
        dataset.add_vector(temp_vec);
    }

    std::cout << "Dataset successfully built in memory!\n\n";

    // Let's create a test query.
    // To PROVE our search works accurately, let's search for an exact copy 
    // of vector #4321 already in our dataset!
    const float* query_vector = dataset.get_vector(4321);

    std::cout << "Searching for nearest neighbor to vector #4321...\n";

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the search
    auto [best_idx, min_dist] = brute_force_search(dataset, query_vector);

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // Output results
    std::cout << "----------------------------------------\n";
    std::cout << "Nearest Neighbor Index : " << best_idx << "\n";
    std::cout << "Expected Index         : 4321\n";
    std::cout << "Distance (Squared)     : " << min_dist << " (Expected ~0.0)\n";
    std::cout << "Search Time            : " << duration.count() << " ms\n";
    std::cout << "----------------------------------------\n";

    return 0;
}