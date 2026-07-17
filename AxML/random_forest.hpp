#pragma once

#ifndef AXML_RANDOM_FOREST_HPP
#define AXML_RANDOM_FOREST_HPP

#include "decision_tree.hpp"
#include <algorithm>

namespace AxML {

    class RandomForestClassifier final : public Classifier {
    public:
        explicit RandomForestClassifier(
            const int32_t n_estimators = 100, const int32_t max_depth = -1,
            const int32_t min_samples_split = 2, const int32_t min_samples_leaf = 1,
            const MaxFeatures max_features = MaxFeatures::Sqrt, const int32_t n_threads = -1, const bool bootstrap = true, const int32_t bootstrap_size = -1,
            const uint64_t random_state = 42)
            : random_state_(random_state), bootstrap_size_(bootstrap_size), n_trees_(n_estimators),
              max_depth_(max_depth), min_samples_split_(min_samples_split),
              min_samples_leaf_(min_samples_leaf), n_threads_(n_threads), max_features_(max_features), bootstrap_(bootstrap) {}

        void fit(const MatrixR &X, const Vector &y) override {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            n_samples_ = X.rows();
            d_ = X.cols();
            if (bootstrap_size_ < 1 || bootstrap_size_ > n_samples_) bootstrap_size_ = n_samples_;

            const int32_t n_workers = std::max(n_threads_, static_cast<int32_t>(std::thread::hardware_concurrency()));
            const int32_t chunk_size = static_cast<int32_t>(std::ceil(static_cast<float>(n_trees_) / static_cast<float>(n_workers)));

            auto [Xc, y_enc, classes] = DecisionTreeClassifier::prepare_shared_data(X, y);
            labels = std::make_shared<std::vector<Scalar>>(classes);

            for (int32_t i = 0; i < n_trees_; ++i) {
                trees_.emplace_back(max_depth_, min_samples_split_, min_samples_leaf_, max_features_, random_state_ + i);
            }

            std::vector<std::jthread> workers(n_workers);

            for (int32_t i = 0; i < n_workers; ++i) {
                const size_t start_idx = i * chunk_size;
                if (start_idx >= n_trees_) break;
                const size_t end_idx = std::min(start_idx + chunk_size, static_cast<size_t>(n_trees_));

                workers.emplace_back([this, start_idx, end_idx, &Xc, &y_enc]() {
                    for (size_t t = start_idx; t < end_idx; ++t) {
                        ll::tl_vec<int32_t> indices;

                        if (bootstrap_) {
                            std::mt19937 gen(random_state_ + t);
                            std::uniform_int_distribution<int32_t> dist(0, bootstrap_size_ - 1);
                            indices.reserve(bootstrap_size_);
                            for (int32_t j = 0; j < bootstrap_size_; ++j) {
                                indices[j] = dist(gen);
                            }
                        }
                        else {
                            indices.reserve(n_samples_);
                            std::iota(indices.begin(), indices.end(), 0);
                        }

                        trees_[t].fit_shared(Xc, y_enc, labels, std::move(indices));
                    }
                });
            }
        }

        Vector predict(const MatrixR &X) const override {
            if (trees_.size() == 0) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR preds = MatrixR::Zero(X.rows(), labels->size());
            for (const auto& tree : trees_) {
                tree.predict_proba_aggregate(X, preds);
            }
            Vector arg_max(X.rows());
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index max_col_idx;
                preds.row(i).maxCoeff(&max_col_idx);
                arg_max(i) = static_cast<Scalar>(max_col_idx);
            }
            return arg_max;
        }

        MatrixR predict_proba(const MatrixR &X) const override {
            if (trees_.size() == 0) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR preds = MatrixR::Zero(X.rows(), labels->size());
            for (const auto& tree : trees_) {
                tree.predict_proba_aggregate(X, preds);
            }
            preds /= static_cast<Scalar>(trees_.size());
            return preds;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override;

        std::string name() const override { return "RandomForestClassifier"; }
        uint32_t type_id() const override { return ID_RF_CLASSIFIER; }

        bool is_fitted() const override {
            return n_trees_ > 0;
        }
        uint64_t dims() const override {
            if (trees_.empty()) {
                return 0;
            }
            return trees_[0].dims();
        }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

    private:
        std::vector<DecisionTreeClassifier> trees_;
        std::shared_ptr<std::vector<Scalar>> labels;
        uint64_t random_state_;
        int32_t n_samples_{};
        int32_t d_{};
        int32_t bootstrap_size_{};
        int32_t n_trees_;
        int32_t max_depth_;
        int32_t min_samples_split_;
        int32_t min_samples_leaf_;
        int32_t n_threads_;
        MaxFeatures max_features_;
        bool bootstrap_;
    };


    class RandomForestRegressor final : public Regressor {
    public:
        explicit RandomForestRegressor(
            const int32_t n_estimators = 100, const int32_t max_depth = -1,
            const int32_t min_samples_split = 2, const int32_t min_samples_leaf = 1,
            const MaxFeatures max_features = MaxFeatures::Sqrt, const int32_t n_threads = -1, const bool bootstrap = true, const int32_t bootstrap_size = -1,
            const uint64_t random_state = 42)
            : random_state_(random_state), bootstrap_size_(bootstrap_size), n_trees_(n_estimators),
              max_depth_(max_depth), min_samples_split_(min_samples_split),
              min_samples_leaf_(min_samples_leaf), n_threads_(n_threads), max_features_(max_features), bootstrap_(bootstrap) {}

        void fit(const MatrixR &X, const Vector &y) override {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            n_samples_ = X.rows();
            if (bootstrap_size_ < 1 || bootstrap_size_ > n_samples_) bootstrap_size_ = n_samples_;

            const int32_t n_workers = std::max(n_threads_, static_cast<int32_t>(std::thread::hardware_concurrency()));
            const int32_t chunk_size = static_cast<int32_t>(std::ceil(static_cast<float>(n_trees_) / static_cast<float>(n_workers)));

            MatrixC Xc = X;

            for (int32_t i = 0; i < n_trees_; ++i) {
                trees_.emplace_back(max_depth_, min_samples_split_, min_samples_leaf_, max_features_, random_state_ + i);
            }

            std::vector<std::jthread> workers(n_workers);

            for (int32_t i = 0; i < n_workers; ++i) {
                const size_t start_idx = i * chunk_size;
                if (start_idx >= n_trees_) break;
                const size_t end_idx = std::min(start_idx + chunk_size, static_cast<size_t>(n_trees_));

                workers.emplace_back([this, start_idx, end_idx, &Xc, &y]() {
                    for (size_t t = start_idx; t < end_idx; ++t) {
                        ll::tl_vec<int32_t> indices;

                        if (bootstrap_) {
                            std::mt19937 gen(random_state_ + t);
                            std::uniform_int_distribution<int32_t> dist(0, bootstrap_size_ - 1);
                            indices.reserve(bootstrap_size_);
                            for (int32_t j = 0; j < bootstrap_size_; ++j) {
                                indices[j] = dist(gen);
                            }
                        }
                        else {
                            indices.reserve(n_samples_);
                            std::iota(indices.begin(), indices.end(), 0);
                        }

                        trees_[t].fit_shared(Xc, y, std::move(indices));
                    }
                });
            }
        }

        Vector predict(const MatrixR &X) const override {
            if (trees_.size() == 0) {
                throw std::runtime_error("Model not fitted yet!");
            }
            Vector preds = Vector::Zero(X.rows());
            for (const auto& tree : trees_) {
                tree.predict_aggregate(X, preds);
            }
            preds /= static_cast<Scalar>(trees_.size());
            return preds;
        }

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override;

        std::string name() const override { return "RandomForestRegressor"; }
        uint32_t type_id() const override { return ID_RF_REGRESSION; }

        bool is_fitted() const override {
            return n_trees_ > 0;
        }
        uint64_t dims() const override {
            if (trees_.empty()) {
                return 0;
            }
            return trees_[0].dims();
        }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            // Do nothing
        }

    private:
        std::vector<DecisionTreeRegressor> trees_;
        uint64_t random_state_;
        int32_t n_samples_{};
        int32_t bootstrap_size_{};
        int32_t n_trees_;
        int32_t max_depth_;
        int32_t min_samples_split_;
        int32_t min_samples_leaf_;
        int32_t n_threads_;
        MaxFeatures max_features_;
        bool bootstrap_;
    };

}

#endif //AXML_RANDOM_FOREST_HPP