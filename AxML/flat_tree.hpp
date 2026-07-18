#pragma once
#ifndef AXML_FLAT_TREE_HPP
#define AXML_FLAT_TREE_HPP

#include <span>
#include "common.hpp"
#include "mem_mgr.hpp"

namespace AxML::detail {

        struct FrozenTree {
            std::vector<i32> feature_;
            std::vector<Scalar> threshold_;
            std::vector<i32> children_left_;
            std::vector<i32> children_right_;
            std::vector<Scalar> value_;
            i32 n_classes_;

            std::span<const Scalar> leaf_value(const i32 node_id) const {
                return { value_.data() + static_cast<size_t>(node_id) * n_classes_, static_cast<size_t>(n_classes_) };
            }
        };

        struct GTree {
            ll::tl_vec<i32> feature_;
            ll::tl_vec<Scalar> threshold_;
            ll::tl_vec<i32> children_left_;
            ll::tl_vec<i32> children_right_;
            ll::tl_vec<Scalar> value_;
            i32 n_classes_;

            explicit GTree(const i32 n_classes = 0) : n_classes_(n_classes) {}

            i32 new_node() {
                feature_.push_back(-2);
                threshold_.push_back(0.0);
                children_left_.push_back(-1);
                children_right_.push_back(-1);
                value_.resize(value_.size() + static_cast<size_t>(n_classes_), 0.0);
                return static_cast<i32>(feature_.size()) - 1;
            }

            void make_leaf(const i32 node_id, const ll::tl_vec<Scalar>& proba) {
                std::ranges::copy(proba, value_.begin() + static_cast<ptrdiff_t>(node_id) * n_classes_);
            }

            void make_leaf(const i32 node_id, const Scalar scalar) {
                if (n_classes_ != 1) {
                    throw std::logic_error("Cannot assign a single scalar to a leaf node when n_classes_ > 1");
                }
                value_.at(static_cast<size_t>(node_id)) = scalar;
            }

            void make_split(const i32 node_id, const i32 feature,
                const Scalar threshold, const i32 left_id, const i32 right_id)
            {
                feature_[node_id] = feature;
                threshold_[node_id] = threshold;
                children_left_[node_id] = left_id;
                children_right_[node_id] = right_id;
            }

            std::span<const Scalar> leaf_value(const i32 node_id) const {
                return { value_.data() + static_cast<size_t>(node_id) * n_classes_, static_cast<size_t>(n_classes_) };
            }

            // Copy the thread local memory back in to global heap space
            FrozenTree freeze() {
                return FrozenTree{
                    std::vector<i32>(feature_.begin(), feature_.end()),
                    std::vector<Scalar>(threshold_.begin(), threshold_.end()),
                    std::vector<i32>(children_left_.begin(), children_left_.end()),
                    std::vector<i32>(children_right_.begin(), children_right_.end()),
                    std::vector<Scalar>(value_.begin(), value_.end()),
                    n_classes_
                };
            }
        };


    // In-place partition of a node's row-subset span
    inline i32 partition_span(std::span<i32> idx, const MatrixC& Xc, const i32 feature, const Scalar threshold) {
        const auto mid = std::partition(idx.begin(), idx.end(),
            [&](const i32 row) { return Xc(row, feature) <= threshold; });
        return static_cast<i32>(mid - idx.begin());
    }
}

#endif //AXML_FLAT_TREE_HPP