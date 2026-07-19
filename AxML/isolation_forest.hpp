
// Deep Isolation Forest implementation is based on the following paper.
//
// @article{Xu_2023,
//    title={Deep Isolation Forest for Anomaly Detection},
//    volume={35},
//    ISSN={2326-3865},
//    url={http://dx.doi.org/10.1109/TKDE.2023.3270293},
//    DOI={10.1109/tkde.2023.3270293},
//    number={12},
//    journal={IEEE Transactions on Knowledge and Data Engineering},
//    publisher={Institute of Electrical and Electronics Engineers (IEEE)},
//    author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
//    year={2023},
//    month=Dec, pages={12591–12604} }

#pragma once

#ifndef AXML_ISOLATION_FOREST_HPP
#define AXML_ISOLATION_FOREST_HPP

#include <random>
#include "common.hpp"
#include "base.hpp"
#include "flat_tree.hpp"

namespace AxML {

    namespace detail {

        struct ITree {

            void fit(const MatrixC& Xc, const std::span<i32> idx, std::mt19937_64& rgen) {
                build_node(idx, 0, Xc, rgen);
                ftree_ = tree_.freeze();
            }

            i32 build_node(const std::span<i32> idx, const i32 depth, const MatrixC& Xc, std::mt19937_64& rgen) {
                const i32 node_id = tree_.new_node();
                const i32 n_node = idx.size();

                std::uniform_int_distribution<i32> feature_dist(0, Xc.cols() - 1);
                Scalar min = std::numeric_limits<Scalar>::max();
                Scalar max = std::numeric_limits<Scalar>::min();
                const i32 rand_feature = feature_dist(rgen);

                bool is_pure = true;
                for (const auto i : idx) {
                    is_pure &= scmp(Xc(i, rand_feature) ,Xc(0, rand_feature));
                }

                if (is_pure || n_node < 2 || (max_depth_ >= 0 && depth >= max_depth_)) {
                    tree_.make_leaf(node_id, static_cast<Scalar>(idx.size()));
                    return node_id;
                }

                for (const auto i : idx) {
                    const Scalar val = Xc(i, rand_feature);
                    if (val < min) min = val;
                    if (val > max) max = val;
                }

                std::uniform_real_distribution<Scalar> real_dist(min, max);
                const Scalar threshold = real_dist(rgen);

                const i32 mid = partition_span(idx, Xc, rand_feature, threshold);
                const auto left_id = build_node(idx.subspan(0, mid), depth + 1, Xc, rgen);
                const auto right_id = build_node(idx.subspan(mid), depth + 1, Xc, rgen);
                tree_.make_split(node_id, rand_feature, threshold, left_id, right_id);
                return node_id;
            }

            template<typename Derived>
            void score_aggregate(const Eigen::DenseBase<Derived>& X, MatrixC& scores) const {
                for (i32 row = 0; row < X.rows(); ++row) {
                    i32 node = 0;
                    Scalar sumDepth = 0.0, sumAbsDiff = 0.0;
                    while (ftree_.children_left_[node] != -1) {
                        sumDepth += 1.0;
                        sumAbsDiff += std::abs(X(row, ftree_.feature_[node]) - ftree_.threshold_[node]);
                        node = X(row, ftree_.feature_[node]) <= ftree_.threshold_[node] ? ftree_.children_left_[node] : ftree_.children_right_[node];
                    }
                    scores(row, 0) += sumDepth + C(ftree_.value_[node]);
                    scores(row, 1) += sumAbsDiff;
                }
            }

            static constexpr Scalar C(const Scalar n) {
                if (n < 2.1) return 0.0;
                if (n < 3.1) return 1.0;
                return 2.0 * (std::log(n - 1.0) + 0.57721566490153286060651209) - (2.0 * (n - 1.0) / n);
            }

            GTree tree_;
            FrozenTree ftree_;
            i32 max_depth_;
        };

    }

    class DeepIsolationForest final : public Transformer {
    public:
        explicit DeepIsolationForest(const i32 representations = 48,
            const i32 sample_size = 256, const i32 trees_per_rep = 6,
            const uint64_t random_state = 42, const bool get_raw_scores = true)
            : random_state_(random_state),
              r_dims_(0),
              representations_(representations),
              sample_size_(sample_size),
              trees_per_rep_(trees_per_rep),
                get_raw_scores_(get_raw_scores)
        {}

        MatrixR transform(const MatrixR &X) const override {
            if ( !is_fitted_ ) {
                throw std::runtime_error("Model not fitted yet!");
            }
            const i32 clvl = std::thread::hardware_concurrency();
            const i32 batch_size = static_cast<i32>(std::ceil(static_cast<Scalar>(X.rows()) / clvl));

            MatrixR final_scores(X.rows(), 1);
            const Scalar CT = detail::ITree::C(sample_size_);

            // This loop will be parallelized
            for (i32 batch = 0; batch < clvl; ++batch) {
                const i32 start_row = batch * batch_size;
                const i32 end_row = std::min(start_row + batch_size, static_cast<i32>(X.rows()));
                const i32 actual_batch_size = end_row - start_row;
                if (start_row >= end_row) break;

                MatrixC scores(actual_batch_size, 2);
                const auto subX = X.middleRows(start_row, actual_batch_size);
                for (const auto& tree : trees_) {
                    tree.score_aggregate(subX, scores);
                }
                scores /= static_cast<Scalar>(trees_.size());
                final_scores.col(0).segment(start_row, actual_batch_size) = (-scores.col(0).array() / CT).array().exp2() * scores.col(1).array();
                Vector anomaly_scores = (-scores.col(0).array() / CT).array().exp2() * scores.col(1).array();
            }

            if (!get_raw_scores_) {
                final_scores = final_scores.unaryExpr([&](const double x) { return x >= 0.7 ? 1.0 : 0.0; });
            }
            return final_scores;
        }

        MatrixR fit_transform(const MatrixR &X) override {
            fit(X, Vector()); // Fit with dummy labels
            return transform(X);
        }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

    public:
        void fit(const MatrixR &X, const Vector &y) override {
            const i32 m = X.cols();
            r_dims_ = m < 20 ? m : std::min(m, 32);
            sample_size_ = std::min(sample_size_, static_cast<i32>(X.rows()));
            const i32 max_depth_ = static_cast<i32>(std::ceil(std::log2(static_cast<double>(sample_size_))));

            std::mt19937_64 gen(random_state_);
            std::uniform_real_distribution<Scalar> dis(0.0, 1.0);

            Vector p = Vector::Zero(m);
            Vector q = Vector::Zero(r_dims_);
            const MatrixC W0 = MatrixC::NullaryExpr(m, r_dims_, [&](){ return dis(gen); });
            std::vector<MatrixC> Ws(representations_);

            for (i32 i = 0; i < representations_; ++i) {
                p = p.unaryExpr([&](double x) { return dis(gen); });
                q = q.unaryExpr([&](double x) { return dis(gen); });
                Ws[i] = W0.cwiseProduct(p * q.transpose());
            }

            const i32 buf_size = sample_size_ * trees_per_rep_;
            MatrixC Xc(buf_size, m);
            std::vector<i32> indices(buf_size);
            const std::span<i32> full_idx_span(indices);

            std::uniform_int_distribution<i32> dist(0, sample_size_ - 1);
            for (i32 rep = 0; rep < representations_; ++rep) {
                gen.seed(random_state_ + rep);
                for (i32 i = 0; i < buf_size; ++i) {
                    const i32 idx = dist(gen);
                    indices[i] = idx;
                    Xc.row(i) = 1.0 / (1.0 + (-(Ws[rep] * X.row(idx)).array()).exp());
                }

                for (i32 t = 0; t < trees_per_rep_; ++t) {
                    trees_.emplace_back();
                    trees_.back().max_depth_ = max_depth_;
                    trees_.back().fit(Xc, full_idx_span.subspan(t * sample_size_, sample_size_), gen);
                }
            }

            is_fitted_ = true;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override {
            return std::make_unique<DeepIsolationForest>(*this);
        }

        bool is_fitted() const override { return true; }

        uint64_t dims() const override { return 0; }

        void reset() override {}

        std::string name() const override { return "IsolationForest"; }

        uint32_t type_id() const override {return ID_ISOLATION_FOREST; }

    private:
        std::vector<detail::ITree> trees_;
        uint64_t random_state_;
        i32 r_dims_;
        i32 representations_;
        i32 sample_size_;
        i32 trees_per_rep_;
        bool get_raw_scores_;
        bool is_fitted_ = false;
    };


}


#endif //AXML_ISOLATION_FOREST_HPP