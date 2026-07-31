#pragma once
#ifndef AXML_DECISION_TREE_HPP
#define AXML_DECISION_TREE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <execution>
#include <random>
#include <ranges>
#include <span>
#include <vector>
#include "base.hpp"
#include "flat_tree.hpp"

namespace AxML {
    namespace detail {
        template<typename T>
        using vec = std::vector<T>;

        struct SplitResult {
            Scalar weighted_impurity = std::numeric_limits<Scalar>::infinity();
            Scalar threshold = 0.0;
            i32 feature = -1;
            bool found = false;
        };

        template<typename T>
        void argsort_and_gather(const MatrixC& Xc, const i32 feature,
                               const std::span<const T> y_enc, const std::span<const i32> idx,
                               std::pmr::vector<Scalar>& xs, std::pmr::vector<T>& ys,
                               std::pmr::memory_resource* pool_res)
        {
            const i32 n = static_cast<i32>(idx.size());
            std::pmr::vector<i32> order(n, pool_res);
            std::iota(order.begin(), order.end(), 0);
            // stable may cause global mutex locks. Using std::sort to maintain no global allocation constraint.
            // even though std::sort is undeterministic
            std::ranges::sort(order, [&](const i32 a, const i32 b) {
                return Xc(idx[a], feature) < Xc(idx[b], feature);
            });
            for (i32 k = 0; k < n; ++k) {
                const i32 row = idx[order[k]];
                xs[k] = Xc(row, feature);
                ys[k] = y_enc[row];
            }
        }

        /*inline SortedColumnScalar argsort_and_gather_scalar(const MatrixC& Xc, const i32 feature,
                                     const Vector& y_enc, const std::span<const i32> idx) {
            const i32 n = static_cast<i32>(idx.size());
            ll::tl_vec<i32> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&](const i32 a, const i32 b) {
                return Xc(idx[a], feature) < Xc(idx[b], feature);
            });

            SortedColumnScalar out;
            out.xs.resize(n);
            out.ys.resize(n);
            for (i32 k = 0; k < n; ++k) {
                const i32 row = idx[order[k]];
                out.xs[k] = Xc(row, feature);
                out.ys[k] = y_enc(row);
            }
            return out;
        }*/

        // Partial Fisher-Yates. After any call, all_features is still a permutation
        // of [0, d), so no re-seeding/reallocation is needed between calls.
        inline void sample_features_inplace(std::vector<i32>& all_features, const i32 k, std::mt19937& rng) {
            const i32 d = static_cast<i32>(all_features.size());
            for (i32 i = 0; i < k; ++i) {
                std::uniform_int_distribution<i32> dist(i, d - 1);
                std::swap(all_features[i], all_features[dist(rng)]);
            }
        }

        inline i32 resolve_max_features(const i32 d, const i32 setting) {
            switch (setting) {
                case -1:  return d; // All
                case -2: // Sqrt
                    return std::max(1, static_cast<i32>(std::sqrt(static_cast<double>(d))));
                case -3: // Log2
                    return std::max(1, static_cast<i32>(std::log2(static_cast<double>(d))));
                default:
                    // any other value: treat as a literal feature count
                    return std::clamp(setting, 1, d);
            }
        }

    }

    class RandomForestClassifier;
    class RandomForestRegressor;

    enum class MaxFeatures : int32_t { All = -1, Sqrt = -2, Log2 = -3 };

    class DecisionTreeClassifier : public Classifier {
    template<typename T>
    using vec = std::vector<T>;

    public:
        explicit DecisionTreeClassifier(
            const i32 max_depth = -1, const i32 min_samples_split = 2,
            const i32 min_samples_leaf = 1, const MaxFeatures max_features = MaxFeatures::Sqrt,
            const uint64_t random_state = 42) :
        max_depth_(max_depth),
        min_samples_split_(min_samples_split),
        min_samples_leaf_(min_samples_leaf),
        max_features_(static_cast<i32>(max_features)),
        random_state_(random_state)
        {}

        void fit(const MatrixR& X, const VectorI& y) override {
            auto [Xc, y_enc, encoder] = prepare_shared_data(X, y);
            std::array<std::byte, 8192> local_buffer;
            std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
            std::pmr::unsynchronized_pool_resource async_res(&mbr);

            std::vector<i32> indices(static_cast<size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), 0);

            detail::TreeCapacityParams tcp;
            tcp.maxDepth = max_depth_;
            tcp.minSamplesLeaf = min_samples_leaf_;
            tcp.minSamplesSplit = min_samples_split_;
            tcp.sampleCount = y.size();

            const auto est_nodes = detail::estimate_tree_capacity(tcp);

            fit_shared(Xc, y_enc, std::make_shared<LabelEncoderInternal>(encoder), std::move(indices), est_nodes, &async_res);
        }

        // Unsafe methods. These do not check for the validity of the inputs or the state of the model
        // NOT RECOMMENDED FOR PUBLIC USE
        i32 predict_label(const MatrixR& X, const i32 row) const {
            auto proba = tree_.get_leaf_value(predict_node(X, row));
            const auto best = std::ranges::max_element(proba);
            return encoder_->inverse_transform(static_cast<i32>(best - proba.begin()));
        }

        i32 predict_label(const Eigen::RowVectorX<Scalar>& X) const {
            auto proba = tree_.get_leaf_value(predict_node(X));
            const auto best = std::ranges::max_element(proba);
            return encoder_->inverse_transform(static_cast<i32>(best - proba.begin()));
        }

        VectorI predict(const MatrixR& X) const override {
            if (X.cols() != dims()) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", d_, X.cols()));
            }
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            VectorI output(X.rows());
            for (i32 i = 0; i < X.rows(); ++i) {
                output(i) = predict_label(X, i);
            }
            return output;
        }

        MatrixR predict_proba(const MatrixR& X) const override {
            if (X.cols() != dims()) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", d_, X.cols()));
            }
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR proba(X.rows(), n_classes_);
            for (i32 i = 0; i < X.rows(); ++i) {
                auto node_proba = tree_.get_leaf_value(predict_node(X, i));
                proba.row(i) = Eigen::Map<const Eigen::RowVectorX<Scalar>>(node_proba.data(), n_classes_);
                //std::memcpy(proba.row(i).data(), node_proba.data(), static_cast<size_t>(n_classes_) * sizeof(Scalar));
            }
            return proba;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override {
            tree_.reset();
            all_features_.clear();
            sample_indices_.clear();
            fitted_ = false;
        }

        std::string name() const override { return "DecisionTreeClassifier"; }
        uint32_t type_id() const override { return ID_DT_CLASSIFIER; }

        i32 n_nodes() const { return tree_.get_num_nodes(); }
        uint64_t dims() const override {return d_;}
        bool is_fitted() const override {return fitted_;}
    private:
        friend RandomForestClassifier;

        const LabelEncoderInternal &get_encoder_() const override {
            return *encoder_;
        }

        void fit_impl(const MatrixR &X, const VectorI &y) override {
            // Do nothing
        }

        void fit_shared(const MatrixC& Xc, const vec<i32>& y_enc,
            const std::shared_ptr<LabelEncoderInternal> &encoder,
            std::vector<i32> initial_indices, const size_t est_nodes,
            std::pmr::memory_resource* pool_res) {

            d_ = static_cast<i32>(Xc.cols());
            encoder_ = encoder;
            n_classes_ = encoder_->num_unique_labels();

            sample_indices_ = std::move(initial_indices);
            n_samples_ = static_cast<i32>(sample_indices_.size());

            all_features_ = std::vector<i32>(d_);
            std::iota(all_features_.begin(), all_features_.end(), 0);

            k_features_ = detail::resolve_max_features(d_, max_features_);
            rng_.seed(random_state_);
            tree_.initialize(est_nodes, n_classes_);

            build_node(std::span<i32>(sample_indices_), 0, Xc, y_enc, pool_res);

            tree_.freeze();
            fitted_ = true;
        }

        static std::tuple<MatrixC, vec<i32>, LabelEncoderInternal> prepare_shared_data(const MatrixR& X, const VectorI& y) {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            LabelEncoderInternal encoder_internal;
            const auto span_y = std::span(y.data(), y.size());
            encoder_internal.fit(span_y);
            vec<i32> y_enc = encoder_internal.transform(span_y);

            MatrixC Xc = X;
            return {std::move(Xc), std::move(y_enc), std::move(encoder_internal)};
        }

        i32 predict_node(const MatrixR& X, const i32 row) const {
            i32 node = 0;
            while (tree_.get_left_child_index(node) != -1) {
                node = X(row, tree_.get_feature_index(node)) <= tree_.get_node_threshold(node)
                           ? tree_.get_left_child_index(node)
                           : tree_.get_right_child_index(node);
            }
            return node;
        }

        i32 predict_node(const Eigen::RowVectorX<Scalar>& X) const {
            i32 node = 0;
            while (tree_.get_left_child_index(node) != -1) {
                node = (X(tree_.get_feature_index(node)) <= tree_.get_node_threshold(node)) ? tree_.get_left_child_index(node) : tree_.get_right_child_index(node);
            }
            return node;
        }

        void predict_proba_aggregate(const MatrixR& X, MatrixR& preds) const {
            for (i32 i = 0; i < X.rows(); ++i) {
                auto node_proba = tree_.get_leaf_value(predict_node(X, i));
                preds.row(i) += Eigen::Map<const Eigen::RowVectorX<double>, Eigen::Unaligned>(node_proba.data(), n_classes_);
            }
        }

        i32 build_node(const std::span<i32> idx, const i32 depth, const MatrixC& Xc, const vec<i32>& y_enc, std::pmr::memory_resource* pool_res) {
            const i32 node_id = tree_.new_node();
            const i32 n_node = static_cast<i32>(idx.size());

            std::pmr::vector<i32> counts(n_classes_, 0, pool_res);
            for (const i32 row : idx) ++counts[y_enc[row]];

            std::pmr::vector<Scalar> proba(n_classes_, 0, pool_res);
            for (i32 c = 0; c < n_classes_; ++c) {
                proba[c] = static_cast<Scalar>(counts[c]) / static_cast<Scalar>(n_node);
            }

            const bool is_pure = (*std::ranges::max_element(counts)) == n_node;
            const bool hit_depth = (max_depth_ >= 0) && (depth >= max_depth_);
            if (n_node < min_samples_split_ || is_pure || hit_depth) {
                tree_.make_leaf(node_id, proba);
                return node_id;
            }

            detail::sample_features_inplace(all_features_, k_features_, rng_);
            detail::SplitResult best;
            for (i32 f = 0; f < k_features_; ++f) {
                const auto chosen_f = all_features_[f];
                auto res = best_split_classification(Xc, chosen_f, y_enc, idx, n_classes_, min_samples_leaf_, pool_res);
                if (res.found && (!best.found || res.weighted_impurity < best.weighted_impurity)) {
                    best = res;
                }
            }

            if (!best.found) {
                tree_.make_leaf(node_id, proba);
                return node_id;
            }

            const i32 mid = detail::partition_span(idx, Xc, best.feature, best.threshold);
            const auto left_id = build_node(idx.subspan(0, mid), depth + 1, Xc, y_enc, pool_res);
            const auto right_id = build_node(idx.subspan(mid), depth + 1, Xc, y_enc, pool_res);
            tree_.make_split(node_id, best.feature, best.threshold, left_id, right_id);
            return node_id;
        }

    static detail::SplitResult best_split_classification(const MatrixC& Xc, const i32 feature, const vec<i32>& y_enc,
                                                         const std::span<const i32> idx, const i32 n_classes,
                                                         const i32 min_samples_leaf, std::pmr::memory_resource* pool_res)
        {
            const std::span y_enc_span{y_enc.data(), y_enc.size()};
            std::pmr::vector<Scalar> xs(idx.size(), pool_res);
            std::pmr::vector<i32> ys_sorted(idx.size(), pool_res);

            detail::argsort_and_gather(Xc, feature, y_enc_span, idx, xs, ys_sorted, pool_res);
            const i32 n = static_cast<i32>(idx.size());

            std::pmr::vector<i32> count_left(n_classes, 0, pool_res), count_right(n_classes, 0, pool_res);
            for (const i32 c : ys_sorted) ++count_right[c];

            Scalar sumsq_left = 0.0, sumsq_right = 0.0;
            for (i32 k = 0; k < n_classes; ++k) {
                sumsq_right += static_cast<Scalar>(count_right[k]) * count_right[k];
            }

            detail::SplitResult best;
            for (i32 i = 0; i < n - 1; ++i) {
                const auto c = ys_sorted[i];
                const i32 old_cl = count_left[c];
                const i32 old_cr = count_right[c];
                sumsq_left  += 2.0 * old_cl + 1.0;
                sumsq_right -= 2.0 * old_cr - 1.0;
                count_left[c]  = old_cl + 1;
                count_right[c] = old_cr - 1;

                if (xs[i] == xs[i + 1]) continue;
                const i32 n_left = i + 1, n_right = n - n_left;
                if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

                const Scalar gini_left  = 1.0 - sumsq_left  / (static_cast<Scalar>(n_left)  * n_left);
                const Scalar gini_right = 1.0 - sumsq_right / (static_cast<Scalar>(n_right) * n_right);
                const Scalar weighted = (n_left * gini_left + n_right * gini_right) / static_cast<Scalar>(n);

                if (weighted < best.weighted_impurity) {
                    best = {weighted, (xs[i] + xs[i + 1]) * 0.5, feature, true};
                }
            }
            return best;
        }

        detail::GTree tree_;
        std::shared_ptr<LabelEncoderInternal> encoder_;
        std::vector<i32> sample_indices_;   // shared permutation buffer, partitioned in place
        std::vector<i32> all_features_;     // reused across every split, zero per-node allocation
        std::mt19937 rng_;

        i32 n_samples_ = 0, d_ = 0, n_classes_ = 0, k_features_ = 0;
        i32 max_depth_, min_samples_split_, min_samples_leaf_, max_features_;
        uint64_t random_state_;
        bool fitted_ = false;
    };

    class DecisionTreeRegressor : public Regressor {
    template<typename T>
    using vec = std::vector<T>;

    public:
        explicit DecisionTreeRegressor(
            const i32 max_depth = -1, const i32 min_samples_split = 2,
            const i32 min_samples_leaf = 1, const MaxFeatures max_features = MaxFeatures::Sqrt,
            const uint64_t random_state = 42) :
        max_depth_(max_depth),
        min_samples_split_(min_samples_split),
        min_samples_leaf_(min_samples_leaf),
        max_features_(static_cast<i32>(max_features)),
        random_state_(random_state)
        {}

        void fit(const MatrixR& X, const Vector& y) override {
            std::array<std::byte, 8192> local_buffer;
            std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
            std::pmr::unsynchronized_pool_resource async_res(&mbr);

            std::vector<i32> indices(static_cast<size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), 0);

            detail::TreeCapacityParams tcp;
            tcp.maxDepth = max_depth_;
            tcp.minSamplesLeaf = min_samples_leaf_;
            tcp.minSamplesSplit = min_samples_split_;
            tcp.sampleCount = y.size();

            const auto est_nodes = detail::estimate_tree_capacity(tcp);

            const MatrixC Xc = X;
            fit_shared(Xc, y, std::move(indices), est_nodes, &async_res);
        }

        Vector predict(const MatrixR& X) const override {
            if (X.cols() != d_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", d_, X.cols()));
            }
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            Vector output(X.rows());
            for (i32 i = 0; i < X.rows(); ++i) {
                output(i) = tree_.get_leaf_value(predict_node(X, i))[0];
            }
            return output;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override {
            tree_.reset();
            all_features_.clear();
            sample_indices_.clear();
            fitted_ = false;
        }

        std::string name() const override { return "DecisionTreeRegressor"; }
        uint32_t type_id() const override { return ID_DT_REGRESSION; }

        i32 n_nodes() const { return tree_.get_num_nodes(); }
        uint64_t dims() const override {return d_;}
        bool is_fitted() const override {return fitted_;}

    private:
        friend RandomForestRegressor;

        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

        void fit_shared(const MatrixC& Xc, const Vector& y, std::vector<i32> initial_indices, const size_t est_nodes, std::pmr::memory_resource* pool_res) {
            if (Xc.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }
            d_ = static_cast<i32>(Xc.cols());

            sample_indices_ = std::move(initial_indices);
            n_samples_ = static_cast<i32>(sample_indices_.size());

            all_features_ = std::vector<i32>(d_);
            std::iota(all_features_.begin(), all_features_.end(), 0);

            k_features_ = detail::resolve_max_features(d_, max_features_);
            rng_.seed(random_state_);
            tree_.initialize(est_nodes, 1); // regression has only one output

            build_node(std::span<i32>(sample_indices_), 0, Xc, y, pool_res);

            tree_.freeze();
            fitted_ = true;
        }

        i32 predict_node(const MatrixR& X, const i32 row) const {
            i32 node = 0;
            while (tree_.get_left_child_index(node) != -1) {
                node = (X(row, tree_.get_feature_index(node)) <= tree_.get_node_threshold(node))
                           ? tree_.get_left_child_index(node)
                           : tree_.get_right_child_index(node);
            }
            return node;
        }

        void predict_aggregate(const MatrixR& X, Vector& preds) const {
            for (i32 i = 0; i < X.rows(); ++i) {
                preds(i) += tree_.get_leaf_value(predict_node(X, i))[0];
            }
        }

        i32 build_node(const std::span<i32> idx, const i32 depth, const MatrixC& Xc, const Vector& y_enc, std::pmr::memory_resource* async_pool) {
            const i32 node_id = tree_.new_node();
            const i32 n_node = static_cast<i32>(idx.size());
            Scalar mean = 0.0;
            bool is_pure = true;
            for (const auto i : idx) {
                mean += y_enc(i);
                is_pure &= scmp(y_enc(i), y_enc(0));
            }
            mean /= static_cast<Scalar>(idx.size());
            const bool hit_depth = (max_depth_ >= 0) && (depth >= max_depth_);
            if (n_node < min_samples_split_ || is_pure || hit_depth) {
                tree_.make_leaf(node_id, mean);
                return node_id;
            }

            detail::sample_features_inplace(all_features_, k_features_, rng_);
            detail::SplitResult best;
            for (i32 f = 0; f < k_features_; ++f) {
                const auto chosen_f = all_features_[f];
                auto res = best_split_regression(Xc, chosen_f, y_enc, idx, min_samples_leaf_, async_pool);
                if (res.found && (!best.found || res.weighted_impurity < best.weighted_impurity)) {
                    best = res;
                }
            }

            if (!best.found) {
                tree_.make_leaf(node_id, mean);
                return node_id;
            }

            const i32 mid = detail::partition_span(idx, Xc, best.feature, best.threshold);
            const auto left_id = build_node(idx.subspan(0, mid), depth + 1, Xc, y_enc, async_pool);
            const auto right_id = build_node(idx.subspan(mid), depth + 1, Xc, y_enc, async_pool);
            tree_.make_split(node_id, best.feature, best.threshold, left_id, right_id);
            return node_id;
        }


        static detail::SplitResult best_split_regression(const MatrixC &Xc, const i32 feature, const Vector &y_enc,
                                                         const std::span<const i32> idx, const i32 min_samples_leaf,
                                                         std::pmr::memory_resource* pool_res)
        {
            const std::span y_enc_span{y_enc.data(), static_cast<size_t>(y_enc.size())};

            std::pmr::vector<Scalar> xs(idx.size(), pool_res);
            std::pmr::vector<Scalar> ys_sorted(idx.size(), pool_res);

            detail::argsort_and_gather(Xc, feature, y_enc_span, idx, xs, ys_sorted, pool_res);

            const i32 n = static_cast<i32>(xs.size());

            const Scalar total_sum = std::transform_reduce(
                std::execution::unseq,
                ys_sorted.begin(), ys_sorted.end(),
                Scalar{0.0},
                std::plus<Scalar>(),
                [](const Scalar x) {return x;}
            );
            const Scalar total_sq = std::transform_reduce(
                std::execution::unseq,
                ys_sorted.begin(), ys_sorted.end(),
                Scalar{0.0},
                std::plus<Scalar>(),
                [](const Scalar x) {return x * x;}
            );


            detail::SplitResult best;
            Scalar running_cum_sum = ys_sorted[0], running_cum_sq = ys_sorted[0] * ys_sorted[0];
            for (i32 i = 0; i < n - 1; ++i) {
                const i32 n_left = i + 1, n_right = n - n_left;

                const Scalar sum_left = running_cum_sum;
                const Scalar sum_sq_left = running_cum_sq;
                const Scalar sum_right = total_sum - sum_left;
                const Scalar sum_sq_right = total_sq - sum_sq_left;
                running_cum_sum += ys_sorted[i + 1];
                running_cum_sq += ys_sorted[i + 1] * ys_sorted[i + 1];

                if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;
                if (xs[i] == xs[i + 1]) continue;

                const Scalar mse_left = (sum_sq_left / n_left) - (sum_left / n_left) * (sum_left / n_left);
                const Scalar mse_right = (sum_sq_right / n_right) - (sum_right / n_right) * (sum_right / n_right);
                const Scalar weighted_mse = (n_left * mse_left + n_right * mse_right) / static_cast<Scalar>(n);

                if (weighted_mse < best.weighted_impurity) {
                    best = {weighted_mse, (xs[i] + xs[i + 1]) * 0.5, feature, true};
                }

            }
            return best;
        }

        detail::GTree tree_; // regression has only one output
        std::vector<i32> sample_indices_;   // shared permutation buffer, partitioned in place
        std::vector<i32> all_features_;     // reused across every split, zero per-node allocation
        std::mt19937 rng_;

        i32 n_samples_ = 0, d_ = 0, k_features_ = 0;
        i32 max_depth_, min_samples_split_, min_samples_leaf_, max_features_;
        uint64_t random_state_;
        bool fitted_ = false;
    };
}

#endif //AXML_DECISION_TREE_HPP