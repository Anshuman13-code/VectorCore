#include "vectorcore/hnsw.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace vectorcore {

// --- HNSWNode Implementation ---
HNSWNode::HNSWNode(NodeId node_id, int max_level, std::size_t m0, std::size_t m)
    : id(node_id), level(max_level), neighbors(max_level + 1) {
    neighbors[0].reserve(m0 + 1);
    for (int i = 1; i <= max_level; ++i) {
        neighbors[i].reserve(m + 1);
    }
}

// --- FlatVectorDataset Implementation ---
FlatVectorDataset::FlatVectorDataset(std::size_t dim) : dimension_(dim) {}
void FlatVectorDataset::reserve(std::size_t num_vectors) { data_.reserve(num_vectors * dimension_); }
void FlatVectorDataset::add_vector(const std::vector<float>& vec) {
    if (dimension_ == 0) dimension_ = vec.size();
    if (vec.size() != dimension_) throw std::invalid_argument("Vector dimension mismatch");
    data_.insert(data_.end(), vec.begin(), vec.end());
}
const float* FlatVectorDataset::get_vector(std::size_t index) const { return data_.data() + (index * dimension_); }
std::size_t FlatVectorDataset::size() const { return dimension_ == 0 ? 0 : data_.size() / dimension_; }
std::size_t FlatVectorDataset::dimension() const { return dimension_; }

void FlatVectorDataset::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + filename);
    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    std::size_t num_elements = data_.size();
    out.write(reinterpret_cast<const char*>(&num_elements), sizeof(num_elements));
    out.write(reinterpret_cast<const char*>(data_.data()), num_elements * sizeof(float));
}

void FlatVectorDataset::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading: " + filename);
    in.read(reinterpret_cast<char*>(&dimension_), sizeof(dimension_));
    std::size_t num_elements;
    in.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
    data_.resize(num_elements);
    in.read(reinterpret_cast<char*>(data_.data()), num_elements * sizeof(float));
}

// --- VisitedTracker Implementation ---
VisitedTracker::VisitedTracker(std::size_t init_capacity) : visited_(init_capacity, 0), current_tag_(0) {}
void VisitedTracker::ensure_capacity(std::size_t size) {
    if (size > visited_.size()) visited_.resize(std::max(size, visited_.size() * 2 + 16), 0);
}
void VisitedTracker::next_query() {
    current_tag_++;
    if (current_tag_ == 0) {
        std::fill(visited_.begin(), visited_.end(), 0);
        current_tag_ = 1;
    }
}
bool VisitedTracker::is_visited(NodeId id) const { return id < visited_.size() && visited_[id] == current_tag_; }
void VisitedTracker::mark_visited(NodeId id) { visited_[id] = current_tag_; }

// --- HNSWIndex Implementation ---
HNSWIndex::HNSWIndex(const FlatVectorDataset& dataset, std::size_t M, std::size_t ef_construction, 
                     std::size_t ef_search, std::size_t expected_size)
    : M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search),
      ml_(1.0 / std::log(M)), max_level_(-1), enter_node_(0), dataset_(dataset),
      insert_scratch_(expected_size), rng_(42) {
    nodes_.reserve(expected_size);
}

int HNSWIndex::get_random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    if (r == 0.0) r = 1e-7;
    return static_cast<int>(-std::log(r) * ml_);
}

void HNSWIndex::prune_neighbors_heuristic(std::vector<NodeId>& neighbors, std::size_t max_conns, const float* base_vec) {
    if (neighbors.size() <= max_conns) return;
    const std::size_t dim = dataset_.dimension();
    std::vector<DistNode> candidates;
    candidates.reserve(neighbors.size());
    for (NodeId n : neighbors) candidates.push_back({l2_distance(base_vec, dataset_.get_vector(n), dim), n});
    std::sort(candidates.begin(), candidates.end());

    std::vector<NodeId> selected;
    selected.reserve(max_conns);

    for (const auto& cand : candidates) {
        if (selected.size() >= max_conns) break;
        const float* cand_vec = dataset_.get_vector(cand.id);
        bool keep = true;
        for (NodeId sel : selected) {
            if (l2_distance(cand_vec, dataset_.get_vector(sel), dim) < cand.dist) {
                keep = false;
                break;
            }
        }
        if (keep) selected.push_back(cand.id);
    }
    for (const auto& cand : candidates) {
        if (selected.size() >= max_conns) break;
        if (std::find(selected.begin(), selected.end(), cand.id) == selected.end()) selected.push_back(cand.id);
    }
    neighbors = std::move(selected);
}

// Updated to use the passed scratchpad
std::vector<DistNode> HNSWIndex::search_layer(const float* query, NodeId ep, std::size_t ef, int layer, SearchScratchpad& scratch) const {
    const std::size_t dim = dataset_.dimension();
    scratch.visited.next_query();

    float ep_dist = l2_distance(query, dataset_.get_vector(ep), dim);
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>> candidates;
    std::priority_queue<DistNode, std::vector<DistNode>, std::less<DistNode>> best_elements;

    candidates.push({ep_dist, ep});
    best_elements.push({ep_dist, ep});
    scratch.visited.mark_visited(ep);

    while (!candidates.empty()) {
        DistNode curr = candidates.top();
        candidates.pop();
        if (curr.dist > best_elements.top().dist) break;

        for (NodeId neighbor : nodes_[curr.id].neighbors[layer]) {
            if (!scratch.visited.is_visited(neighbor)) {
                scratch.visited.mark_visited(neighbor);
                float d = l2_distance(query, dataset_.get_vector(neighbor), dim);
                if (d < best_elements.top().dist || best_elements.size() < ef) {
                    candidates.push({d, neighbor});
                    best_elements.push({d, neighbor});
                    if (best_elements.size() > ef) best_elements.pop();
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

NodeId HNSWIndex::search_upper_layers(const float* query, NodeId ep, int from_level, int to_level) const {
    const std::size_t dim = dataset_.dimension();
    NodeId curr = ep;
    float curr_dist = l2_distance(query, dataset_.get_vector(curr), dim);
    for (int lc = from_level; lc >= to_level; --lc) {
        bool improved = true;
        while (improved) {
            improved = false;
            NodeId best_neighbor = curr;
            float best_dist = curr_dist;
            for (NodeId neighbor : nodes_[curr].neighbors[lc]) {
                float d = l2_distance(query, dataset_.get_vector(neighbor), dim);
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

void HNSWIndex::insert(NodeId id) {
    if (id != nodes_.size()) throw std::runtime_error("Node IDs must be sequential 0..N-1");

    int insert_level = get_random_level();
    nodes_.emplace_back(id, insert_level, M0_, M_);
    insert_scratch_.resize(nodes_.size());

    if (max_level_ == -1) {
        max_level_ = insert_level;
        enter_node_ = id;
        return;
    }

    const float* query = dataset_.get_vector(id);
    NodeId curr = enter_node_;

    if (insert_level < max_level_) {
        curr = search_upper_layers(query, curr, max_level_, insert_level + 1);
    }

    int start_level = std::min(max_level_, insert_level);
    for (int lc = start_level; lc >= 0; --lc) {
        // Pass internal insert_scratch_ to search_layer
        auto candidates = search_layer(query, curr, ef_construction_, lc, insert_scratch_);

        std::size_t forward_conns = M_;
        std::vector<NodeId> cand_ids;
        cand_ids.reserve(candidates.size());
        for (const auto& c : candidates) cand_ids.push_back(c.id);

        prune_neighbors_heuristic(cand_ids, forward_conns, query);
        nodes_[id].neighbors[lc] = cand_ids;

        std::size_t max_reverse = (lc == 0) ? M0_ : M_;
        for (NodeId neighbor : nodes_[id].neighbors[lc]) {
            nodes_[neighbor].neighbors[lc].push_back(id);
            prune_neighbors_heuristic(nodes_[neighbor].neighbors[lc], max_reverse, dataset_.get_vector(neighbor));
        }
        if (!candidates.empty()) curr = candidates.front().id;
    }
    if (insert_level > max_level_) {
        max_level_ = insert_level;
        enter_node_ = id;
    }
}

void HNSWIndex::set_ef_search(std::size_t ef) { ef_search_ = ef; }

// Updated to use the passed scratchpad
std::vector<DistNode> HNSWIndex::query(const float* query_vec, std::size_t K, SearchScratchpad& scratch) const {
    if (nodes_.empty()) return {};
    NodeId curr = enter_node_;
    if (max_level_ > 0) {
        curr = search_upper_layers(query_vec, curr, max_level_, 1);
    }
    auto layer0_candidates = search_layer(query_vec, curr, std::max(ef_search_, K), 0, scratch);
    if (layer0_candidates.size() > K) layer0_candidates.resize(K);
    return layer0_candidates;
}

void HNSWIndex::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open index file for writing: " + filename);
    out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
    out.write(reinterpret_cast<const char*>(&M0_), sizeof(M0_));
    out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
    out.write(reinterpret_cast<const char*>(&ef_search_), sizeof(ef_search_));
    out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
    out.write(reinterpret_cast<const char*>(&enter_node_), sizeof(enter_node_));

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

void HNSWIndex::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open index file for reading: " + filename);
    in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
    in.read(reinterpret_cast<char*>(&M0_), sizeof(M0_));
    in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
    in.read(reinterpret_cast<char*>(&ef_search_), sizeof(ef_search_));
    in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
    in.read(reinterpret_cast<char*>(&enter_node_), sizeof(enter_node_));
    ml_ = 1.0 / std::log(M_);

    std::size_t num_nodes;
    in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    nodes_.clear();
    nodes_.reserve(num_nodes);
    
    // Resize the write scratchpad so it's ready for future insertions
    insert_scratch_.resize(num_nodes);

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

int HNSWIndex::max_level() const { return max_level_; }
std::size_t HNSWIndex::size() const { return nodes_.size(); }

} // namespace vectorcore