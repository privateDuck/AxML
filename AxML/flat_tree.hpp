#pragma once
#ifndef AXML_FLAT_TREE_HPP
#define AXML_FLAT_TREE_HPP

#include <span>
#include "common.hpp"
#include <memory_resource>

namespace AxML::detail {

    struct GTree {
        std::vector<i32> values_i32;    // feature (i + 0), c_left (i + 1), c_right (i + 2)
        std::vector<Scalar> values_scalar; // threshold (i + 0), proba (i + 1 to i + 1 + n_classes)

        i32 n_classes_;
        i32 num_nodes_;

        explicit GTree() : n_classes_(0), num_nodes_(0) {}

        explicit GTree(const size_t estimated_nodes, const i32 n_classes = 0)
            : n_classes_(n_classes), num_nodes_(0)
        {
            // Preallocate buffers to the estimated node count
            values_i32.resize(estimated_nodes * 3, 0);
            values_scalar.resize(estimated_nodes * (n_classes_ + 1), 0.0);
        }

        void initialize(const size_t estimated_nodes, const i32 n_classes) {
            n_classes_ = n_classes;
            values_i32.resize(estimated_nodes * 3, 0);
            values_scalar.resize(estimated_nodes * (n_classes_ + 1), 0.0);
        }

        i32 new_node() {
            const size_t node_idx = num_nodes_++;

            // Check bounds and resize. Although this should never occur, as the
            // estimated count is greater than the maximum possible nodes
            if (node_idx * 3 + 2 >= values_i32.size()) {
                // Double the size
                values_i32.resize(values_i32.size() * 2);
                values_scalar.resize(values_scalar.size() * 2);
            }

            // Initialize default values
            values_i32[node_idx * 3 + 0] = -2; // feature
            values_i32[node_idx * 3 + 1] = -1; // children_left
            values_i32[node_idx * 3 + 2] = -1; // children_right

            values_scalar[node_idx * (n_classes_ + 1) + 0] = 0.0; // threshold
            return static_cast<i32>(node_idx);
        }

        void make_leaf(const i32 node_id, const std::vector<Scalar>& proba) {
            std::ranges::copy(proba, values_scalar.begin() + static_cast<ptrdiff_t>(node_id) * (n_classes_ + 1) + 1);
        }

        void make_leaf(const i32 node_id, const std::pmr::vector<Scalar>& proba) {
            std::ranges::copy(proba, values_scalar.begin() + static_cast<ptrdiff_t>(node_id) * (n_classes_ + 1) + 1);
        }

        void make_leaf(const i32 node_id, const Scalar scalar) {
            if (n_classes_ != 1) {
                throw std::logic_error("Cannot assign a single scalar to a leaf node when n_classes_ > 1");
            }
            values_scalar[static_cast<size_t>(node_id) * (n_classes_ + 1) + 1] = scalar;
        }

        void make_split(const i32 node_id, const i32 feature,
            const Scalar threshold, const i32 left_id, const i32 right_id)
        {
            values_i32[static_cast<size_t>(node_id) * 3 + 0] = feature;
            values_i32[static_cast<size_t>(node_id) * 3 + 1] = left_id;
            values_i32[static_cast<size_t>(node_id) * 3 + 2] = right_id;
            values_scalar[static_cast<size_t>(node_id) * (n_classes_ + 1) + 0] = threshold;
        }

        i32 get_feature_index(const i32 node_id) const {
            return values_i32[static_cast<size_t>(node_id) * 3 + 0];
        }

        i32 get_left_child_index(const i32 node_id) const {
            return values_i32[static_cast<size_t>(node_id) * 3 + 1];
        }

        i32 get_right_child_index(const i32 node_id) const {
            return values_i32[static_cast<size_t>(node_id) * 3 + 2];
        }

        std::span<const Scalar> get_leaf_value(const i32 node_id) const {
            return { values_scalar.data() + static_cast<size_t>(node_id) * (n_classes_ + 1) + 1, static_cast<size_t>(n_classes_) };
        }

        Scalar get_node_threshold(const i32 node_id) const {
            return values_scalar[static_cast<size_t>(node_id) * (n_classes_ + 1) + 0];
        }

        i32 get_num_nodes() const {
            return num_nodes_;
        }

        void reset() {
            // Just reset the node counter to allow reuse of allocated memory
            num_nodes_ = 0;
        }

        void freeze() {
            values_i32.resize(static_cast<size_t>(num_nodes_) * 3);
            values_scalar.resize(static_cast<size_t>(num_nodes_) * (n_classes_ + 1));

            values_i32.shrink_to_fit();
            values_scalar.shrink_to_fit();
        }
    };

    // In-place partition of a node's row-subset span
    inline i32 partition_span(std::span<i32> idx, const MatrixC& Xc, const i32 feature, const Scalar threshold) {
        const auto mid = std::partition(idx.begin(), idx.end(),
            [&](const i32 row) { return Xc(row, feature) <= threshold; });
        return static_cast<i32>(mid - idx.begin());
    }

    struct TreeCapacityParams
    {
        std::size_t sampleCount;

        int maxDepth = -1;                 // -1 = unlimited
        std::size_t minSamplesSplit = 2;
        std::size_t minSamplesLeaf = 1;
        std::size_t maxLeafNodes = 0;      // 0 = unlimited

        // Tunable heuristics
        double leafFactor = 1.5;           // Avg leaf ≈ 1.5 * minSamplesLeaf
        double splitFactor = 0.75;         // Avg leaf ≈ 0.75 * minSamplesSplit
        double fillFactor = 0.60;          // Typical occupancy
        double overallocation = 1.50;      // Safety margin
    };

    inline std::size_t estimate_tree_capacity(const TreeCapacityParams& p)
    {
        if (p.sampleCount == 0)
            return 0;

        const double avgLeafSize =
            std::max(p.leafFactor * static_cast<double>(p.minSamplesLeaf),
                p.splitFactor * static_cast<double>(p.minSamplesSplit));

        double estimatedLeaves = static_cast<double>(p.sampleCount) / avgLeafSize;

        // max_depth constraint
        if (p.maxDepth >= 0)
        {
            estimatedLeaves = std::min(
                estimatedLeaves,
                std::ldexp(1.0, p.maxDepth)); // 2^maxDepth
        }

        // max_leaf_nodes constraint
        if (p.maxLeafNodes > 0)
        {
            estimatedLeaves = std::min(estimatedLeaves, static_cast<double>(p.maxLeafNodes));
        }

        // Binary tree node count
        double estimatedNodes = (2.0 * estimatedLeaves - 1.0) * p.fillFactor;

        estimatedNodes *= p.overallocation;

        return std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(estimatedNodes)));
    }

    inline std::size_t estimate_max_node_count(
        const std::size_t sampleCount,
        const int maxDepth,
        const std::size_t minSamplesLeaf,
        const std::size_t maxLeafNodes)
    {
        std::size_t maxLeaves = sampleCount;

        if (maxDepth >= 0)
            maxLeaves = std::min(maxLeaves, std::size_t{1} << maxDepth);

        maxLeaves = std::min(maxLeaves, sampleCount / minSamplesLeaf);

        if (maxLeafNodes > 0)
            maxLeaves = std::min(maxLeaves, maxLeafNodes);

        return maxLeaves == 0 ? 1 : (2 * maxLeaves - 1);
    }
}

#endif //AXML_FLAT_TREE_HPP