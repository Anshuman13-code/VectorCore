#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <random>
#include <chrono>
#include <cmath>
#include <queue>
#include <algorithm>
#include <arm_neon.h>


using NodeId = uint32_t;

// -------------------------------------------------------------
// 1. ARM NEON SIMD Distance Kernel
// -------------------------------------------------------------
inline float l2_neon(const float* a, const float* b, std::size_t dim) {
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
}

// -------------------------------------------------------------
// 2. Continuous Flat Dataset
// -------------------------------------------------------------
class FlatVectorDataset {
private:
    std::size_t dimension_;
    std::vector<float> data_;

public:
    explicit FlatVectorDataset(std::size_t dim) : dimension_(dim) {}

    void reserve(std::size_t num_vectors) {
        data_.reserve(num_vectors * dimension_);
    }

    void add_vector(const std::vector<float>& vec) {
        data_.insert(data_.end(), vec.begin(), vec.end());
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

    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open file for writing: " + filename);

        out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
        std::size_t num_elements = data_.size();
        out.write(reinterpret_cast<const char*>(&num_elements), sizeof(num_elements));
        out.write(reinterpret_cast<const char*>(data_.data()), num_elements * sizeof(float));
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open file for reading: " + filename);

        in.read(reinterpret_cast<char*>(&dimension_), sizeof(dimension_));
        std::size_t num_elements;
        in.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        
        data_.resize(num_elements);
        in.read(reinterpret_cast<char*>(data_.data()), num_elements * sizeof(float));
    }
};

// -------------------------------------------------------------
// 3. Distance Pairs and HNSW Node
// -------------------------------------------------------------
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

    HNSWNode(NodeId node_id, int max_level, std::size_t m0, std::size_t m)
        : id(node_id), level(max_level), neighbors(max_level + 1) {
        neighbors[0].reserve(m0 + 1);
        for (int i = 1; i <= max_level; ++i) {
            neighbors[i].reserve(m + 1);
        }
    }
};

// -------------------------------------------------------------
// 4. O(1) Amortized Visited Tracker (Zero O(N^2) allocations)
// -------------------------------------------------------------
class VisitedTracker {
private:
    std::vector<uint32_t> visited_;
    uint32_t current_tag_;

public:
    VisitedTracker(std::size_t init_capacity = 0) : visited_(init_capacity, 0), current_tag_(0) {}

    void ensure_capacity(std::size_t size) {
        if (size > visited_.size()) {
            visited_.resize(std::max(size, visited_.size() * 2 + 16), 0);
        }
    }

    void next_query() {
        current_tag_++;
        if (current_tag_ == 0) { // On 32-bit overflow
            std::fill(visited_.begin(), visited_.end(), 0);
            current_tag_ = 1;
        }
    }

    bool is_visited(NodeId id) const {
        return id < visited_.size() && visited_[id] == current_tag_;
    }

    void mark_visited(NodeId id) {
        visited_[id] = current_tag_;
    }
};

// -------------------------------------------------------------
// 5. Full Production HNSW Index
// -------------------------------------------------------------
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
    mutable VisitedTracker visited_tracker_;
    std::mt19937 rng_;

    int get_random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng_);
        if (r == 0.0) r = 1e-7;
        return static_cast<int>(-std::log(r) * ml_);
    }

    // Heuristic neighbor pruning with diversity selection (Algorithm 4 in HNSW)
    void prune_neighbors_heuristic(std::vector<NodeId>& neighbors, std::size_t max_conns, const float* base_vec) {
        if (neighbors.size() <= max_conns) return;

        const std::size_t dim = dataset_.dimension();
        std::vector<DistNode> candidates;
        candidates.reserve(neighbors.size());

        for (NodeId n : neighbors) {
            candidates.push_back({l2_neon(base_vec, dataset_.get_vector(n), dim), n});
        }
        std::sort(candidates.begin(), candidates.end()); // Sort ascending by distance to base

        std::vector<NodeId> selected;
        selected.reserve(max_conns);

        for (const auto& cand : candidates) {
            if (selected.size() >= max_conns) break;
            const float* cand_vec = dataset_.get_vector(cand.id);

            bool keep = true;
            for (NodeId sel : selected) {
                float dist_to_selected = l2_neon(cand_vec, dataset_.get_vector(sel), dim);
                // Diversity constraint: discard if candidate is closer to an already-selected neighbor than to base
                if (dist_to_selected < cand.dist) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(cand.id);
            }
        }

        // If diversity pruning was overly strict, backfill with closest candidates to guarantee connectivity
        for (const auto& cand : candidates) {
            if (selected.size() >= max_conns) break;
            if (std::find(selected.begin(), selected.end(), cand.id) == selected.end()) {
                selected.push_back(cand.id);
            }
        }

        neighbors = std::move(selected);
    }

public:
    HNSWIndex(const FlatVectorDataset& dataset, std::size_t M = 16, std::size_t ef_construction = 64, std::size_t ef_search = 64, std::size_t expected_size = 10000)
        : M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search),
          ml_(1.0 / std::log(M)), max_level_(-1), enter_node_(0), dataset_(dataset),
          visited_tracker_(expected_size), rng_(42) {
        nodes_.reserve(expected_size);
    }

    std::vector<DistNode> search_layer(const float* query, NodeId ep, std::size_t ef, int layer) const {
        const std::size_t dim = dataset_.dimension();
        visited_tracker_.next_query();

        float ep_dist = l2_neon(query, dataset_.get_vector(ep), dim);

        std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
        std::priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> best_elements;

        candidates.push({ep_dist, ep});
        best_elements.push({ep_dist, ep});
        visited_tracker_.mark_visited(ep);

        while (!candidates.empty()) {
            DistNode curr = candidates.top();
            candidates.pop();

            float worst_best_dist = best_elements.top().dist;
            if (curr.dist > worst_best_dist) {
                break;
            }

            for (NodeId neighbor : nodes_[curr.id].neighbors[layer]) {
                if (!visited_tracker_.is_visited(neighbor)) {
                    visited_tracker_.mark_visited(neighbor);

                    float d = l2_neon(query, dataset_.get_vector(neighbor), dim);
                    worst_best_dist = best_elements.top().dist;

                    if (d < worst_best_dist || best_elements.size() < ef) {
                        candidates.push({d, neighbor});
                        best_elements.push({d, neighbor});

                        if (best_elements.size() > ef) {
                            best_elements.pop();
                        }
                    }
                }
            }
        }

        std::vector<DistNode> result;
        result.reserve(best_elements.size());
        while (!best_elements.empty()) {
            result.push_back(best_elements.top());
            best_elements.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    NodeId search_upper_layers(const float* query, NodeId ep, int from_level, int to_level) const {
        const std::size_t dim = dataset_.dimension();
        NodeId curr = ep;
        float curr_dist = l2_neon(query, dataset_.get_vector(curr), dim);

        for (int lc = from_level; lc >= to_level; --lc) {
            bool improved = true;
            while (improved) {
                improved = false;
                NodeId best_neighbor = curr;
                float best_dist = curr_dist;

                for (NodeId neighbor : nodes_[curr].neighbors[lc]) {
                    float d = l2_neon(query, dataset_.get_vector(neighbor), dim);
                    if (d < best_dist) {
                        best_dist = d;
                        best_neighbor = neighbor;
                        improved = true;
                    }
                }

                if (improved) {
                    curr = best_neighbor;
                    curr_dist = best_dist;
                }
            }
        }
        return curr;
    }

    void insert(NodeId id) {
        if (id != nodes_.size()) {
            std::cerr << "Error: Node IDs must be sequential.\n";
            std::exit(1);
        }

        int insert_level = get_random_level();
        nodes_.emplace_back(id, insert_level, M0_, M_);
        visited_tracker_.ensure_capacity(nodes_.size());

        if (max_level_ == -1) {
            max_level_ = insert_level;
            enter_node_ = id;
            return;
        }

        const float* query = dataset_.get_vector(id);
        NodeId curr = enter_node_;

        // 1. Greedy descent down to layer (insert_level + 1)
        if (insert_level < max_level_) {
            curr = search_upper_layers(query, curr, max_level_, insert_level + 1);
        }

        // 2. Beam search and edge attachment across layers min(max_level, insert_level) down to 0
        int start_level = std::min(max_level_, insert_level);
        for (int lc = start_level; lc >= 0; --lc) {
            auto candidates = search_layer(query, curr, ef_construction_, lc);

            // Forward connections: select M closest (or M0 for layer 0 if needed, but standard selects M)
            std::size_t forward_conns = M_;
            std::vector<NodeId> cand_ids;
            cand_ids.reserve(candidates.size());
            for (const auto& c : candidates) cand_ids.push_back(c.id);

            prune_neighbors_heuristic(cand_ids, forward_conns, query);
            nodes_[id].neighbors[lc] = cand_ids;

            // Reciprocal connections: prune neighbors with heuristic
            std::size_t max_reverse = (lc == 0) ? M0_ : M_;
            for (NodeId neighbor : nodes_[id].neighbors[lc]) {
                nodes_[neighbor].neighbors[lc].push_back(id);
                prune_neighbors_heuristic(nodes_[neighbor].neighbors[lc], max_reverse, dataset_.get_vector(neighbor));
            }

            if (!candidates.empty()) {
                curr = candidates.front().id;
            }
        }

        if (insert_level > max_level_) {
            max_level_ = insert_level;
            enter_node_ = id;
        }
    }

    void set_ef_search(std::size_t ef) {
        ef_search_ = ef;
    }

    std::vector<DistNode> query(const float* query_vec, std::size_t K) const {
        if (nodes_.empty()) return {};

        NodeId curr = enter_node_;
        if (max_level_ > 0) {
            curr = search_upper_layers(query_vec, curr, max_level_, 1);
        }

        auto layer0_candidates = search_layer(query_vec, curr, std::max(ef_search_, K), 0);
        if (layer0_candidates.size() > K) {
            layer0_candidates.resize(K);
        }
        return layer0_candidates;
    }

    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open index file for writing.");

        // 1. Write global parameters
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&M0_), sizeof(M0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&ef_search_), sizeof(ef_search_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&enter_node_), sizeof(enter_node_));

        // 2. Write graph nodes
        std::size_t num_nodes = nodes_.size();
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

        for (const auto& node : nodes_) {
            out.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
            out.write(reinterpret_cast<const char*>(&node.level), sizeof(node.level));
            
            for (int lc = 0; lc <= node.level; ++lc) {
                std::size_t num_neighbors = node.neighbors[lc].size();
                out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));
                if (num_neighbors > 0) {
                    out.write(reinterpret_cast<const char*>(node.neighbors[lc].data()), num_neighbors * sizeof(NodeId));
                }
            }
        }
    }

    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open index file for reading.");

        // 1. Read global parameters
        in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
        in.read(reinterpret_cast<char*>(&M0_), sizeof(M0_));
        in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
        in.read(reinterpret_cast<char*>(&ef_search_), sizeof(ef_search_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&enter_node_), sizeof(enter_node_));

        ml_ = 1.0 / std::log(M_);

        // 2. Read graph nodes
        std::size_t num_nodes;
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
        nodes_.clear();
        nodes_.reserve(num_nodes);
        visited_tracker_.ensure_capacity(num_nodes);

        for (std::size_t i = 0; i < num_nodes; ++i) {
            NodeId id;
            int level;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&level), sizeof(level));

            nodes_.emplace_back(id, level, M0_, M_);
            
            for (int lc = 0; lc <= level; ++lc) {
                std::size_t num_neighbors;
                in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
                nodes_.back().neighbors[lc].resize(num_neighbors);
                if (num_neighbors > 0) {
                    in.read(reinterpret_cast<char*>(nodes_.back().neighbors[lc].data()), num_neighbors * sizeof(NodeId));
                }
            }
        }
    }

    int max_level() const { return max_level_; }
};

// -------------------------------------------------------------
// 6. Ground-Truth Brute Force Oracle
// -------------------------------------------------------------
std::vector<DistNode> brute_force_knn(const FlatVectorDataset& dataset, const float* query, std::size_t K) {
    std::priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> max_heap;
    const std::size_t total = dataset.size();
    const std::size_t dim = dataset.dimension();

    for (std::size_t i = 0; i < total; ++i) {
        float d = l2_neon(query, dataset.get_vector(i), dim);
        if (max_heap.size() < K) {
            max_heap.push({d, static_cast<NodeId>(i)});
        } else if (d < max_heap.top().dist) {
            max_heap.pop();
            max_heap.push({d, static_cast<NodeId>(i)});
        }
    }

    std::vector<DistNode> res;
    res.reserve(max_heap.size());
    while (!max_heap.empty()) {
        res.push_back(max_heap.top());
        max_heap.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
}

// -------------------------------------------------------------
// 7. Synthetic Clustered Mixture of Gaussians Dataset
// -------------------------------------------------------------
void generate_clustered_data(FlatVectorDataset& train, FlatVectorDataset& test, std::size_t num_train, std::size_t num_test, std::size_t dim, std::size_t num_clusters = 8) {
    std::mt19937 rng(42);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> cluster_dist(0, num_clusters - 1);

    // Generate cluster centroids
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim));
    for (std::size_t c = 0; c < num_clusters; ++c) {
        for (std::size_t d = 0; d < dim; ++d) {
            centroids[c][d] = norm_dist(rng) * 5.0f;
        }
    }

    auto fill_dataset = [&](FlatVectorDataset& ds, std::size_t count) {
        std::vector<float> vec(dim);
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t c = cluster_dist(rng);
            for (std::size_t d = 0; d < dim; ++d) {
                vec[d] = centroids[c][d] + norm_dist(rng) * 0.5f; // Clustered tightly around centroid
            }
            ds.add_vector(vec);
        }
    };

    fill_dataset(train, num_train);
    fill_dataset(test, num_test);
}

// -------------------------------------------------------------
// 8. Benchmark Suite
// -------------------------------------------------------------
int main() {
    constexpr std::size_t DIM = 128;
    constexpr std::size_t NUM_TRAIN = 10000;
    constexpr std::size_t NUM_TEST = 200;
    constexpr std::size_t K = 10;

    std::cout << "--- 1. Generating Clustered GMM Dataset ---\n";
    FlatVectorDataset train_data(DIM);
    FlatVectorDataset test_data(DIM);
    train_data.reserve(NUM_TRAIN);
    test_data.reserve(NUM_TEST);
    generate_clustered_data(train_data, test_data, NUM_TRAIN, NUM_TEST, DIM, 8);

    std::cout << "--- 2. Building Production HNSW Index (M=16, efConstruction=64) ---\n";
    auto t_build_start = std::chrono::high_resolution_clock::now();
    HNSWIndex index(train_data, 16, 64, 64, NUM_TRAIN);
    for (std::size_t i = 0; i < NUM_TRAIN; ++i) {
        index.insert(static_cast<NodeId>(i));
    }
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << "Index built in: " << build_sec << " s | Max Level: " << index.max_level() << "\n\n";

    // Add this right after the "Index built in..." line in main()
    std::cout << "--- Testing Binary Serialization ---\n";
    train_data.save("dataset.bin");
    index.save("index.bin");
    std::cout << "Saved dataset.bin and index.bin to disk.\n";

    // Simulate an engine restart by creating a blank instance
    FlatVectorDataset loaded_data(DIM);
    loaded_data.load("dataset.bin");
    
    HNSWIndex loaded_index(loaded_data, 16, 64, 64, 0); // 0 expected size, will reserve on load
    loaded_index.load("index.bin");
    std::cout << "Successfully loaded engine from disk! Nodes: " << loaded_data.size() << "\n\n";

    // IMPORTANT: Swap out `index` for `loaded_index` in the parameter sweep below this point to test the reloaded graph!

    std::cout << "--- 3. Running Ground-Truth Brute Force Oracle (" << NUM_TEST << " queries) ---\n";
    std::vector<std::vector<DistNode>> ground_truth(NUM_TEST);
    auto t_bf_start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < NUM_TEST; ++q) {
        ground_truth[q] = brute_force_knn(train_data, test_data.get_vector(q), K);
    }
    auto t_bf_end = std::chrono::high_resolution_clock::now();
    double bf_total_ms = std::chrono::duration<double, std::milli>(t_bf_end - t_bf_start).count();
    std::cout << "Brute Force Latency: " << (bf_total_ms / NUM_TEST) << " ms/query | QPS: " << (NUM_TEST / (bf_total_ms / 1000.0)) << "\n\n";

    // Warm-up run to wake up performance cores and warm caches
    loaded_index.set_ef_search(16);
    for (std::size_t q = 0; q < 50; ++q) {
        volatile auto sink = loaded_index.query(test_data.get_vector(q), K);
        (void)sink;
    }

    std::cout << "--- 4. HNSW Parameter Sweep (Isolated Timing & True Diversity Pruning) ---\n";
    std::vector<std::size_t> ef_sweep = {16, 32, 64, 128, 256};
    std::vector<std::vector<DistNode>> query_results(NUM_TEST);

    for (std::size_t ef : ef_sweep) {
        loaded_index.set_ef_search(ef);

        // Benchmark query time ONLY (no evaluation logic inside the timer)
        auto t_start = std::chrono::high_resolution_clock::now();
        for (std::size_t q = 0; q < NUM_TEST; ++q) {
            query_results[q] = loaded_index.query(test_data.get_vector(q), K);
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double mean_latency_ms = total_ms / NUM_TEST;
        double qps = NUM_TEST / (total_ms / 1000.0);

        // Evaluate recall separately outside the timer
        std::size_t total_matches = 0;
        for (std::size_t q = 0; q < NUM_TEST; ++q) {
            for (const auto& res : query_results[q]) {
                for (const auto& gt : ground_truth[q]) {
                    if (res.id == gt.id) {
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