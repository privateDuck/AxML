#pragma once

#ifndef AXML_RANDOM_FOREST_HPP
#define AXML_RANDOM_FOREST_HPP

#include "decision_tree.hpp"
#include <algorithm>
#include <thread>

namespace AxML {

    class RandomForestClassifier final : public Classifier {
    public:
        explicit RandomForestClassifier(const RandomForestParams& params)
            : random_state_(params.random_state), bootstrap_size_(params.bootstrap_size), n_trees_(params.n_estimators),
              max_depth_(params.max_depth), min_samples_split_(params.min_samples_split),
              min_samples_leaf_(params.min_samples_leaf), n_threads_(params.n_threads), max_features_(params.get_mf()), bootstrap_(params.enable_bootstrap) {}

        void fit(const ConstMatRRef &X, const ConstVecIRef &y) override {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            n_samples_ = static_cast<i32>(X.rows());
            d_ = static_cast<i32>(X.cols());
            if (bootstrap_size_ < 1 || bootstrap_size_ > n_samples_) bootstrap_size_ = n_samples_;

            const u32 n_workers = n_threads_ <= 0 ? std::thread::hardware_concurrency() : std::min(n_threads_, static_cast<i32>(std::thread::hardware_concurrency()));
            const u32 chunk_size = static_cast<u32>(std::ceil(static_cast<float>(n_trees_) / static_cast<float>(n_workers)));

            auto [Xc, y_enc, encoder] = DecisionTreeClassifier::prepare_shared_data(X, y);
            encoder_ = std::make_shared<LabelEncoderInternal>(std::move(encoder));

            DecisionTreeParams params;
            params.max_depth = max_depth_;
            params.min_samples_leaf = min_samples_leaf_;
            params.min_samples_split = min_samples_split_;
            params.max_features = DecisionTreeParams::get_mf_str(max_features_);
            params.random_state = random_state_;
            for (int32_t i = 1; i <= n_trees_; ++i) {
                params.random_state += i;
                trees_.emplace_back(params);
            }

            detail::TreeCapacityParams tcp;
            tcp.maxDepth = max_depth_;
            tcp.minSamplesLeaf = min_samples_leaf_;
            tcp.minSamplesSplit = min_samples_split_;
            tcp.sampleCount = y_enc.size();

            const auto est_nodes = detail::estimate_tree_capacity(tcp);

            if (n_threads_ != 0) {

                std::vector<std::jthread> workers;
                workers.reserve(n_workers);
                std::exception_ptr exception_ptr = nullptr;

                for (int32_t i = 0; i < n_workers; ++i) {
                    const size_t start_idx = i * chunk_size;
                    if (start_idx >= n_trees_) break;
                    const size_t end_idx = std::min(start_idx + chunk_size, static_cast<size_t>(n_trees_));

                    workers.emplace_back([this, start_idx, end_idx, est_nodes, &Xc, &y_enc, &exception_ptr]() {
                        try {
                            std::array<std::byte, 8192> local_buffer{};
                            std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
                            std::pmr::unsynchronized_pool_resource async_res(&mbr);

                            for (size_t t = start_idx; t < end_idx; ++t) {
                                std::vector<int32_t> indices;
                                std::mt19937_64 gen(random_state_ + t);

                                if (bootstrap_) {
                                    std::uniform_int_distribution<int32_t> dist(0, n_samples_ - 1);
                                    indices.resize(bootstrap_size_);
                                    for (int32_t j = 0; j < bootstrap_size_; ++j) {
                                        indices[j] = dist(gen);
                                    }
                                }
                                else {
                                    indices.resize(n_samples_);
                                    std::iota(indices.begin(), indices.end(), 0);
                                }

                                trees_[t].fit_shared(Xc, y_enc, encoder_, std::move(indices), est_nodes, &async_res, gen);
                            }
                        }
                        catch (...) {
                            exception_ptr = std::current_exception();
                        }
                    });
                }

                if (exception_ptr) {
                    std::rethrow_exception(exception_ptr);
                }
            }
            else {
                std::array<std::byte, 8192> local_buffer{};
                std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
                std::pmr::unsynchronized_pool_resource async_res(&mbr);

                for (size_t t = 0; t < n_trees_; ++t) {
                    std::vector<int32_t> indices;
                    std::mt19937_64 gen(random_state_ + t);

                    if (bootstrap_) {
                        std::uniform_int_distribution<int32_t> dist(0, n_samples_ - 1);
                        indices.resize(bootstrap_size_);
                        for (int32_t j = 0; j < bootstrap_size_; ++j) {
                            indices[j] = dist(gen);
                        }
                    }
                    else {
                        indices.resize(n_samples_);
                        std::iota(indices.begin(), indices.end(), 0);
                    }

                    trees_[t].fit_shared(Xc, y_enc, encoder_, std::move(indices), est_nodes, &async_res, gen);
                }
            }
        }

        [[nodiscard]] VectorI predict(const ConstMatRRef &X) const override {
            if (X.cols() != d_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", d_, X.cols()));
            }
            if (trees_.empty()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR preds = MatrixR::Zero(X.rows(), encoder_->num_unique_labels());
            for (const auto& tree : trees_) {
                tree.predict_proba_aggregate(X, preds);
            }
            VectorI arg_max(X.rows());
            for (Eigen::Index i = 0; i < X.rows(); ++i) {
                Eigen::Index max_col_idx;
                preds.row(i).maxCoeff(&max_col_idx);
                arg_max(i) = encoder_->inverse_transform(static_cast<i32>(max_col_idx));
            }
            return arg_max;
        }

        [[nodiscard]] MatrixR predict_proba(const ConstMatRRef &X) const override {
            if (X.cols() != d_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", d_, X.cols()));
            }
            if (trees_.empty()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR preds = MatrixR::Zero(X.rows(), encoder_->num_unique_labels());
            for (const auto& tree : trees_) {
                tree.predict_proba_aggregate(X, preds);
            }
            preds /= static_cast<Scalar>(trees_.size());
            return preds;
        }

        void save(OutputArchive &ar) const override{}
        void load(InputArchive &ar) override{}
        [[nodiscard]] std::unique_ptr<Estimator> clone() const override {return std::make_unique<RandomForestClassifier>(*this);}

        void reset() override{trees_.clear();}

        [[nodiscard]] std::string name() const override { return "RandomForestClassifier"; }
        [[nodiscard]] uint32_t type_id() const override { return ID_RF_CLASSIFIER; }

        [[nodiscard]] bool is_fitted() const override {
            return n_trees_ > 0;
        }
        [[nodiscard]] uint64_t dims() const override {
            if (trees_.empty()) {
                return 0;
            }
            return d_;
        }

    protected:
        [[nodiscard]] const LabelEncoderInternal &get_encoder_() const override {
            return *encoder_;
        }

        void fit_impl(const ConstMatRRef &X, const ConstVecIRef &y) override {
            // Do nothing
        }

    private:
        std::vector<DecisionTreeClassifier> trees_;
        std::shared_ptr<LabelEncoderInternal> encoder_;
        uint64_t random_state_;
        int32_t n_samples_{};
        int32_t d_{};
        int32_t bootstrap_size_{};
        int32_t n_trees_;
        int32_t max_depth_;
        int32_t min_samples_split_;
        int32_t min_samples_leaf_;
        int32_t n_threads_;
        i32 max_features_;
        bool bootstrap_;
    };


    class RandomForestRegressor final : public Regressor {
    public:
        explicit RandomForestRegressor(const RandomForestParams& params)
            : random_state_(params.random_state), bootstrap_size_(params.bootstrap_size), n_trees_(params.n_estimators),
              max_depth_(params.max_depth), min_samples_split_(params.min_samples_split),
              min_samples_leaf_(params.min_samples_leaf), n_threads_(params.n_threads), max_features_(params.get_mf()), bootstrap_(params.enable_bootstrap) {}

        void fit(const ConstMatRRef &X, const ConstVecRef &y) override {
            if (X.rows() != y.size()) {
                throw std::invalid_argument("Size mismatch. X.rows() must be equal to y.size()");
            }

            n_samples_ = static_cast<i32>(X.rows());
            if (bootstrap_size_ < 1 || bootstrap_size_ > n_samples_) bootstrap_size_ = n_samples_;

            const u32 n_workers = n_threads_ <= 0 ? std::thread::hardware_concurrency() : std::min(n_threads_, static_cast<i32>(std::thread::hardware_concurrency()));
            const u32 chunk_size = static_cast<u32>(std::ceil(static_cast<float>(n_trees_) / static_cast<float>(n_workers)));

            MatrixC Xc = X;
            DecisionTreeParams params;
            params.max_depth = max_depth_;
            params.min_samples_leaf = min_samples_leaf_;
            params.min_samples_split = min_samples_split_;
            params.max_features = DecisionTreeParams::get_mf_str(max_features_);
            params.random_state = random_state_;
            for (int32_t i = 1; i <= n_trees_; ++i) {
                params.random_state += i;
                trees_.emplace_back(params);
            }

            if (n_threads_ != 0) {
                std::vector<std::jthread> workers(n_workers);
                workers.reserve(n_workers);
                std::exception_ptr exception_ptr = nullptr;

                for (int32_t i = 0; i < n_workers; ++i) {
                    const size_t start_idx = i * chunk_size;
                    if (start_idx >= n_trees_) break;
                    const size_t end_idx = std::min(start_idx + chunk_size, static_cast<size_t>(n_trees_));

                    workers.emplace_back([this, start_idx, end_idx, &Xc, &y, &exception_ptr]() {
                        try {
                            std::array<std::byte, 8192> local_buffer{};
                            std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
                            std::pmr::unsynchronized_pool_resource async_res(&mbr);

                            detail::TreeCapacityParams tcp;
                            tcp.maxDepth = max_depth_;
                            tcp.minSamplesLeaf = min_samples_leaf_;
                            tcp.minSamplesSplit = min_samples_split_;
                            tcp.sampleCount = y.size();

                            const auto est_nodes = detail::estimate_tree_capacity(tcp);

                            for (size_t t = start_idx; t < end_idx; ++t) {
                                std::vector<int32_t> indices;
                                std::mt19937_64 gen(random_state_ + t);

                                if (bootstrap_) {
                                    std::uniform_int_distribution<int32_t> dist(0, n_samples_ - 1);
                                    indices.resize(bootstrap_size_);
                                    for (int32_t j = 0; j < bootstrap_size_; ++j) {
                                        indices[j] = dist(gen);
                                    }
                                }
                                else {
                                    indices.resize(n_samples_);
                                    std::iota(indices.begin(), indices.end(), 0);
                                }

                                trees_[t].fit_shared(Xc, y, std::move(indices), est_nodes, &async_res, gen);
                            }
                        }
                        catch (...) {
                            exception_ptr = std::current_exception();
                        }
                    });
                }

                if (exception_ptr) {
                    std::rethrow_exception(exception_ptr);
                }
            }
            else {
                std::array<std::byte, 8192> local_buffer{};
                std::pmr::monotonic_buffer_resource mbr(local_buffer.data(), local_buffer.size(), std::pmr::new_delete_resource());
                std::pmr::unsynchronized_pool_resource async_res(&mbr);

                detail::TreeCapacityParams tcp;
                tcp.maxDepth = max_depth_;
                tcp.minSamplesLeaf = min_samples_leaf_;
                tcp.minSamplesSplit = min_samples_split_;
                tcp.sampleCount = y.size();

                const auto est_nodes = detail::estimate_tree_capacity(tcp);

                for (size_t t = 0; t < n_trees_; ++t) {
                    std::vector<int32_t> indices;
                    std::mt19937_64 gen(random_state_ + t);

                    if (bootstrap_) {
                        std::uniform_int_distribution<int32_t> dist(0, n_samples_ - 1);
                        indices.resize(bootstrap_size_);
                        for (int32_t j = 0; j < bootstrap_size_; ++j) {
                            indices[j] = dist(gen);
                        }
                    }
                    else {
                        indices.resize(n_samples_);
                        std::iota(indices.begin(), indices.end(), 0);
                    }

                    trees_[t].fit_shared(Xc, y, std::move(indices), est_nodes, &async_res, gen);
                }
            }
        }

        [[nodiscard]] Vector predict(const ConstMatRRef &X) const override {
            if (X.cols() != dims()) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", dims(), X.cols()));
            }
            if (trees_.empty()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            Vector preds = Vector::Zero(X.rows());
            for (const auto& tree : trees_) {
                tree.predict_aggregate(X, preds);
            }
            preds /= static_cast<Scalar>(trees_.size());
            return preds;
        }

        void save(OutputArchive &ar) const override{}
        void load(InputArchive &ar) override{}
        [[nodiscard]] std::unique_ptr<Estimator> clone() const override {return std::make_unique<RandomForestRegressor>(*this);}

        void reset() override{trees_.clear();}

        [[nodiscard]] std::string name() const override { return "RandomForestRegressor"; }
        [[nodiscard]] uint32_t type_id() const override { return ID_RF_REGRESSION; }

        [[nodiscard]] bool is_fitted() const override {
            return n_trees_ > 0;
        }
        [[nodiscard]] uint64_t dims() const override {
            if (trees_.empty()) {
                return 0;
            }
            return trees_[0].dims();
        }

    protected:
        void fit_impl(const ConstMatRRef &X, const ConstVecRef &y) override {
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
        i32 max_features_;
        bool bootstrap_;
    };

}

#endif //AXML_RANDOM_FOREST_HPP