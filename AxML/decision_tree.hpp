#pragma once
#ifndef AXML_DECISION_TREE_HPP
#define AXML_DECISION_TREE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
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

        struct SortedColumn {
            ll::tl_vec<Scalar> xs;
            ll::tl_vec<i32> ys;      // encoded class label, matches xs order
        };

        struct SortedColumnScalar {
            ll::tl_vec<Scalar> xs;
            ll::tl_vec<Scalar> ys;      // encoded class label, matches xs order
        };

        struct SplitResult {
            Scalar weighted_impurity = std::numeric_limits<Scalar>::infinity();
            Scalar threshold = 0.0;
            i32 feature = -1;
            bool found = false;
        };

        inline SortedColumn argsort_and_gather(const MatrixC& Xc, const i32 feature,
                                     const vec<i32>& y_enc, const std::span<const i32> idx) {
            const i32 n = static_cast<i32>(idx.size());
            ll::tl_vec<i32> order(n);
            std::iota(order.begin(), order.end(), 0);
            // stable may cause global mutex locks. Using std::sort to maintain strict no global allocation constraint.
            // even though std::sort is undeterministic
            std::ranges::sort(order, [&](const i32 a, const i32 b) {
                return Xc(idx[a], feature) < Xc(idx[b], feature);
            });

            SortedColumn out;
            out.xs.resize(n);
            out.ys.resize(n);
            for (i32 k = 0; k < n; ++k) {
                const i32 row = idx[order[k]];
                out.xs[k] = Xc(row, feature);
                out.ys[k] = y_enc[row];
            }
            return out;
        }

        inline SortedColumnScalar argsort_and_gather_scalar(const MatrixC& Xc, const i32 feature,
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
        }

        // Partial Fisher-Yates. After any call, all_features is still a permutation
        // of [0, d), so no re-seeding/reallocation is needed between calls.
        inline void sample_features_inplace(ll::tl_vec<i32>& all_features, const i32 k, std::mt19937& rng) {
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

        void fit(const MatrixR& X, const Vector& y) override {
            auto [Xc, y_enc, classes] = prepare_shared_data(X, y);
            ll::tl_vec<i32> indices(static_cast<size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), 0);
            fit_shared(Xc, y_enc, std::make_shared<vec<Scalar>>(classes), std::move(indices));
        }

        Scalar predict_label(const MatrixR& X, const i32 row) const {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            auto proba = ftree_.leaf_value(predict_node(X, row));
            const auto best = std::ranges::max_element(proba);
            return (*classes_)[static_cast<i32>(best - proba.begin())];
        }

        Scalar predict_label(const Eigen::RowVectorX<Scalar>& X) const {
            auto proba = ftree_.leaf_value(predict_node(X));
            const auto best = std::ranges::max_element(proba);
            return (*classes_)[static_cast<i32>(best - proba.begin())];
        }

        Vector predict(const MatrixR& X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            Vector output(X.rows());
            for (i32 i = 0; i < X.rows(); ++i) {
                output(i) = predict_label(X, i);
            }
            return output;
        }

        MatrixR predict_proba(const MatrixR& X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR proba(X.rows(), n_classes_);
            for (i32 i = 0; i < X.rows(); ++i) {
                auto node_proba = ftree_.leaf_value(predict_node(X, i));
                proba.row(i) = Eigen::Map<const Eigen::RowVectorX<Scalar>>(node_proba.data(), n_classes_);
                //std::memcpy(proba.row(i).data(), node_proba.data(), static_cast<size_t>(n_classes_) * sizeof(Scalar));
            }
            return proba;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override {
            tree_ = detail::GTree(0);
            all_features_.clear();
            sample_indices_.clear();
            fitted_ = false;
        }

        std::string name() const override { return "DecisionTreeClassifier"; }
        uint32_t type_id() const override { return ID_DT_CLASSIFIER; }

        i32 n_nodes() const { return static_cast<i32>(tree_.feature_.size()); }
        uint64_t dims() const override {return d_;}
        bool is_fitted() const override {return fitted_;}
    private:
        friend RandomForestClassifier;

        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

        void fit_shared(const MatrixC& Xc, const vec<i32>& y_enc, const std::shared_ptr<vec<Scalar>>& classes, ll::tl_vec<i32> initial_indices) {
            d_ = static_cast<i32>(Xc.cols());
            classes_ = classes;
            n_classes_ = static_cast<i32>(classes_->size());

            sample_indices_ = std::move(initial_indices);
            n_samples_ = static_cast<i32>(sample_indices_.size());

            all_features_.resize(d_);
            std::iota(all_features_.begin(), all_features_.end(), 0);

            k_features_ = detail::resolve_max_features(d_, max_features_);
            rng_.seed(random_state_);
            tree_ = detail::GTree(n_classes_);

            build_node(std::span<i32>(sample_indices_), 0, Xc, y_enc);

            ftree_ = tree_.freeze();
            fitted_ = true;
        }

        static std::tuple<MatrixC, vec<i32>, vec<Scalar>> prepare_shared_data(const MatrixR& X, const Vector& y) {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            vec<Scalar> unique_labels(y.data(), y.data() + y.size());
            std::ranges::sort(unique_labels);
            unique_labels.erase(std::ranges::unique(unique_labels).begin(), unique_labels.end());

            vec<i32> y_enc(static_cast<size_t>(y.size()));
            for (i32 i = 0; i < static_cast<i32>(y.size()); ++i) {
                auto it = std::ranges::lower_bound(unique_labels, y(i));
                y_enc[static_cast<size_t>(i)] = static_cast<i32>(it - unique_labels.begin());
            }
            MatrixC Xc = X;
            return {std::move(Xc), std::move(y_enc), std::move(unique_labels)};
        }

        i32 predict_node(const MatrixR& X, const i32 row) const {
            i32 node = 0;
            while (ftree_.children_left_[node] != -1) {
                node = (X(row, ftree_.feature_[node]) <= ftree_.threshold_[node])
                           ? ftree_.children_left_[node]
                           : ftree_.children_right_[node];
            }
            return node;
        }

        i32 predict_node(const Eigen::RowVectorX<Scalar>& X) const {
            i32 node = 0;
            while (ftree_.children_left_[node] != -1) {
                node = (X(ftree_.feature_[node]) <= ftree_.threshold_[node]) ? ftree_.children_left_[node] : ftree_.children_right_[node];
            }
            return node;
        }

        void predict_proba_aggregate(const MatrixR& X, MatrixR& preds) const {
            for (i32 i = 0; i < X.rows(); ++i) {
                auto node_proba = ftree_.leaf_value(predict_node(X, i));
                preds.row(i) += Eigen::Map<const Eigen::RowVectorX<double>, Eigen::Unaligned>(node_proba.data(), n_classes_);
            }
        }

        i32 build_node(const std::span<i32> idx, const i32 depth, const MatrixC& Xc, const vec<i32>& y_enc) {
            const i32 node_id = tree_.new_node();
            const i32 n_node = static_cast<i32>(idx.size());

            ll::tl_vec<i32> counts(n_classes_, 0);
            for (const i32 row : idx) ++counts[y_enc[row]];

            ll::tl_vec<Scalar> proba(n_classes_, 0);
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
                auto res = best_split_classification(Xc, chosen_f, y_enc, idx, n_classes_, min_samples_leaf_);
                if (res.found && (!best.found || res.weighted_impurity < best.weighted_impurity)) {
                    best = res;
                }
            }

            if (!best.found) {
                tree_.make_leaf(node_id, proba);
                return node_id;
            }

            const i32 mid = detail::partition_span(idx, Xc, best.feature, best.threshold);
            const auto left_id = build_node(idx.subspan(0, mid), depth + 1, Xc, y_enc);
            const auto right_id = build_node(idx.subspan(mid), depth + 1, Xc, y_enc);
            tree_.make_split(node_id, best.feature, best.threshold, left_id, right_id);
            return node_id;
        }

    static detail::SplitResult best_split_classification(const MatrixC& Xc, const i32 feature, const vec<i32>& y_enc,
                                                         const std::span<const i32> idx, const i32 n_classes,
                                                         const i32 min_samples_leaf)
        {
            auto [xs, ys_sorted] = detail::argsort_and_gather(Xc, feature, y_enc, idx);
            const i32 n = static_cast<i32>(xs.size());

            ll::tl_vec<i32> count_left(n_classes, 0), count_right(n_classes, 0);
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

        detail::GTree tree_{0};
        detail::FrozenTree ftree_{};
        std::shared_ptr<vec<Scalar>> classes_;
        ll::tl_vec<i32> sample_indices_;   // shared permutation buffer, partitioned in place
        ll::tl_vec<i32> all_features_;     // reused across every split, zero per-node allocation
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
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }
            const MatrixC Xc = X;
            ll::tl_vec<i32> indices(static_cast<size_t>(X.rows()));
            std::iota(indices.begin(), indices.end(), 0);
            fit_shared(Xc, y, std::move(indices));
            fitted_ = true;
        }

        Vector predict(const MatrixR& X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            Vector output(X.rows());
            for (i32 i = 0; i < X.rows(); ++i) {
                output(i) = ftree_.leaf_value(predict_node(X, i))[0];
            }
            return output;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override {
            tree_ = detail::GTree(0);
            all_features_.clear();
            sample_indices_.clear();
            fitted_ = false;
        }

        std::string name() const override { return "DecisionTreeRegressor"; }
        uint32_t type_id() const override { return ID_DT_REGRESSION; }

        i32 n_nodes() const { return static_cast<i32>(ftree_.feature_.size()); }
        uint64_t dims() const override {return d_;}
        bool is_fitted() const override {return fitted_;}

    private:
        friend RandomForestRegressor;

        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

        void fit_shared(const MatrixC& Xc, const Vector& y_enc, ll::tl_vec<i32> initial_indices) {
            d_ = static_cast<i32>(Xc.cols());

            sample_indices_ = std::move(initial_indices);
            n_samples_ = static_cast<i32>(sample_indices_.size());

            all_features_.resize(d_);
            std::iota(all_features_.begin(), all_features_.end(), 0);

            k_features_ = detail::resolve_max_features(d_, max_features_);
            rng_.seed(random_state_);
            tree_ = detail::GTree(1); // regression has only one output

            build_node(std::span<i32>(sample_indices_), 0, Xc, y_enc);

            ftree_ = tree_.freeze();
        }

        i32 predict_node(const MatrixR& X, const i32 row) const {
            i32 node = 0;
            while (ftree_.children_left_[node] != -1) {
                node = (X(row, ftree_.feature_[node]) <= ftree_.threshold_[node])
                           ? ftree_.children_left_[node]
                           : ftree_.children_right_[node];
            }
            return node;
        }

        void predict_aggregate(const MatrixR& X, Vector& preds) const {
            for (i32 i = 0; i < X.rows(); ++i) {
                preds(i) += ftree_.leaf_value(predict_node(X, i))[0];
            }
        }

        i32 build_node(const std::span<i32> idx, const i32 depth, const MatrixC& Xc, const Vector& y_enc) {
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
                auto res = best_split_regression(Xc, chosen_f, y_enc, idx, min_samples_leaf_);
                if (res.found && (!best.found || res.weighted_impurity < best.weighted_impurity)) {
                    best = res;
                }
            }

            if (!best.found) {
                tree_.make_leaf(node_id, mean);
                return node_id;
            }

            const i32 mid = detail::partition_span(idx, Xc, best.feature, best.threshold);
            const auto left_id = build_node(idx.subspan(0, mid), depth + 1, Xc, y_enc);
            const auto right_id = build_node(idx.subspan(mid), depth + 1, Xc, y_enc);
            tree_.make_split(node_id, best.feature, best.threshold, left_id, right_id);
            return node_id;
        }


        static detail::SplitResult best_split_regression(const MatrixC &Xc, const i32 feature, const Vector &y_enc,
                                                         const std::span<const i32> idx, const i32 min_samples_leaf)
        {
            auto [xs, ys_sorted] = detail::argsort_and_gather_scalar(Xc, feature, y_enc, idx);
            const i32 n = static_cast<i32>(xs.size());

            ll::tl_vec<Scalar> cum_sum(n, 0.0), cum_sq(n, 0.0);
            for (i32 i = 0; i < n; ++i) {
                cum_sum[i] = ys_sorted[i] + (i > 0 ? cum_sum[i - 1] : 0.0);
                cum_sq[i] = ys_sorted[i] * ys_sorted[i] + (i > 0 ? cum_sq[i - 1] : 0.0);
            }
            const Scalar total_sum = cum_sum[n - 1];
            const Scalar total_sq = cum_sq[n - 1];

            detail::SplitResult best;
            for (i32 i = 0; i < n - 1; ++i) {
                if (xs[i] == xs[i + 1]) continue;
                const i32 n_left = i + 1, n_right = n - n_left;
                if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

                const Scalar sum_left = cum_sum[i];
                const Scalar sum_sq_left = cum_sq[i];
                const Scalar sum_right = total_sum - sum_left;
                const Scalar sum_sq_right = total_sq - sum_sq_left;

                const Scalar mse_left = (sum_sq_left / n_left) - (sum_left / n_left) * (sum_left / n_left);
                const Scalar mse_right = (sum_sq_right / n_right) - (sum_right / n_right) * (sum_right / n_right);
                const Scalar weighted_mse = (n_left * mse_left + n_right * mse_right) / static_cast<Scalar>(n);

                if (weighted_mse < best.weighted_impurity) {
                    best = {weighted_mse, (xs[i] + xs[i + 1]) * 0.5, feature, true};
                }
            }
            return best;
        }

        detail::GTree tree_{0};
        detail::FrozenTree ftree_{};
        ll::tl_vec<i32> sample_indices_;   // shared permutation buffer, partitioned in place
        ll::tl_vec<i32> all_features_;     // reused across every split, zero per-node allocation
        std::mt19937 rng_;

        i32 n_samples_ = 0, d_ = 0, k_features_ = 0;
        i32 max_depth_, min_samples_split_, min_samples_leaf_, max_features_;
        uint64_t random_state_;
        bool fitted_ = false;
    };
}

#endif //AXML_DECISION_TREE_HPP