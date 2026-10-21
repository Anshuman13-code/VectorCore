#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include <queue>
#include <random>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace vectorcore {

using NodeId = uint32_t;

inline float l2_distance(const float* a, const float* b, std::size_t dim) {
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }
    float sum = vaddvq_f32(acc);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#else
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
#endif
}

struct DistNode {
    float dist;
    NodeId id;
    bool operator>(const DistNode& other) const { return dist > other.dist; }
    bool operator<(const DistNode& other) const { return dist < other.dist; }
};

struct HNSWNode {
    NodeId id;
    int level;
    std::vector<std::vector<NodeId>> neighbors;
    HNSWNode(NodeId node_id, int max_level, std::size_t m0, std::size_t m);
};

class FlatVectorDataset {
private:
    std::size_t dimension_;
    std::vector<float> data_;
public:
    explicit FlatVectorDataset(std::size_t dim = 0);
    void reserve(std::size_t num_vectors);
    void add_vector(const std::vector<float>& vec);
    const float* get_vector(std::size_t index) const;
    std::size_t size() const;
    std::size_t dimension() const;
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};

class VisitedTracker {
private:
    std::vector<uint32_t> visited_;
    uint32_t current_tag_;
public:
    explicit VisitedTracker(std::size_t init_capacity = 0);
    void ensure_capacity(std::size_t size);
    void next_query();
    bool is_visited(NodeId id) const;
    void mark_visited(NodeId id);
};

// --- NEW: Thread-Local Search Scratchpad ---
struct SearchScratchpad {
    VisitedTracker visited;
    explicit SearchScratchpad(std::size_t size) : visited(size) {}
    void resize(std::size_t size) { visited.ensure_capacity(size); }
};

class HNSWIndex {
private:
    std::size_t M_;
    std::size_t M0_;
    std::size_t ef_construction_;
    std::size_t ef_search_;
    double ml_;
    int max_level_;
    NodeId enter_node_;

    const FlatVectorDataset& dataset_;
    std::vector<HNSWNode> nodes_;
    
    // Used strictly for non-concurrent graph building
    SearchScratchpad insert_scratch_; 
    std::mt19937 rng_;

    int get_random_level();
    void prune_neighbors_heuristic(std::vector<NodeId>& neighbors, std::size_t max_conns, const float* base_vec);
    
    // Notice how search_layer now requires the caller to provide a scratchpad
    std::vector<DistNode> search_layer(const float* query, NodeId ep, std::size_t ef, int layer, SearchScratchpad& scratch) const;
    NodeId search_upper_layers(const float* query, NodeId ep, int from_level, int to_level) const;

public:
    HNSWIndex(const FlatVectorDataset& dataset, std::size_t M = 16, std::size_t ef_construction = 64, 
              std::size_t ef_search = 64, std::size_t expected_size = 10000);

    void insert(NodeId id);
    void set_ef_search(std::size_t ef);
    
    // --- NEW: Query requires a thread-local scratchpad ---
    std::vector<DistNode> query(const float* query_vec, std::size_t K, SearchScratchpad& scratch) const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);
    int max_level() const;
    std::size_t size() const;
};

} // namespace vectorcore