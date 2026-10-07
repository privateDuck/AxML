#pragma once
#ifndef AXML_MLP_HPP
#define AXML_MLP_HPP

#include "base.hpp"
#include "feedforward/feedforward_network.hpp"
#include "feedforward/optimizer.hpp"

namespace AxML {

    class MLPClassifier final : public Classifier {
    public:
        explicit MLPClassifier(MLPParams params) : params_(std::move(params)) {}
        void save(OutputArchive &ar) const override {}

        void load(InputArchive &ar) override {}

        [[nodiscard]] std::unique_ptr<Estimator> clone() const override { return std::make_unique<MLPClassifier>(*this); }

        // TODO: FIX
        [[nodiscard]] bool is_fitted() const override { return is_fitted_; }

        [[nodiscard]] uint64_t dims() const override { return n_features;}

        void reset() override {}

        [[nodiscard]] std::string name() const override {return "MLPClassifier";}

        [[nodiscard]] uint32_t type_id() const override {return ID_MLP_CLASSIFIER;}

        [[nodiscard]] VectorI predict(const ConstMatRRef &X) const override {
            if (X.cols() != dims()) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", dims(), X.cols()));
            }
            if (!is_fitted()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR logits = nn_.forward(X);
            VectorI preds(logits.rows());
            for (i32 i = 0; i < logits.rows(); ++i) {
                Eigen::Index arg;
                logits.row(i).maxCoeff(&arg);
                preds(i) = encoder_.inverse_transform(static_cast<i32>(arg));
            }
            return preds;
        }

        [[nodiscard]] MatrixR predict_proba(const ConstMatRRef &X) const override {
            if (X.cols() != dims()) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", dims(), X.cols()));
            }
            if (!is_fitted()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR logits = nn_.forward(X);
            detail::Softmax softmax;
            return softmax.forward(logits).output;
        }

        [[nodiscard]] bool supports_predict_proba() const noexcept override {return true;}

    protected:
        [[nodiscard]] detail::RegularizationFunc get_reg_fn() const {
            if (params_.regularization == "none") {
                return detail::NoRegularization{};
            }
            if (params_.regularization == "l1") {
                return detail::L1Regularization{};
            }
            if (params_.regularization == "l2") {
                return detail::L2Regularization{};
            }
            if (params_.regularization == "elastic_net") {
                return detail::ElasticNetRegularization{};
            }
            throw std::runtime_error("Regularization function '" + params_.regularization + "' is not supported for MLP");
        }

        [[nodiscard]] detail::ActivationFunc get_act_fn() const {
            if (params_.activation == "relu") {
                return detail::ReLU();
            }
            if (params_.activation == "sigmoid") {
                return detail::Sigmoid();
            }
            if (params_.activation == "tanh") {
                return detail::Tanh();
            }
            if (params_.activation == "linear") {
                return detail::Linear();
            }
            throw std::runtime_error("Activation function '" + params_.activation + "' is not supported for MLP");
        }

        void fit_impl(const ConstMatRRef &X, const ConstVecIRef &y) override {
            encoder_.fit(y);
            n_features = X.cols();
            n_classes = encoder_.num_unique_labels();
            const auto y_one_hot = encoder_.transform_to_onehot_row_major(y);
            const auto y_one_hot_mat = MapConstMatrixR(y_one_hot.data(), static_cast<Eigen::Index>(y_one_hot.size() / n_classes), n_classes);
            const auto act = get_act_fn();
            const auto reg = get_reg_fn();

            i32 in_dim = static_cast<i32>(X.cols());

            for (int out_dim : params_.hidden_layers) {
                nn_.addLayer(in_dim, out_dim, act, reg);

                in_dim = out_dim;
            }

            nn_.addLayer(
                in_dim,
                static_cast<i32>(n_classes),
                detail::Linear{},
                detail::NoRegularization{}
            );

            ff::OptimizerType opt;
            if (params_.solver == "sgd") {
                opt = ff::SGDOptimizer(params_.learning_rate, params_.momentum);
            }
            else if (params_.solver == "adam") {
                opt = ff::AdamOptimizer(params_.learning_rate, params_.beta_1, params_.beta_2, params_.epsilon);
            }
            else {
                throw std::runtime_error("Solver '" + params_.solver + "' is not supported. Only 'sgd' and 'adam' are supported.");
            }

            ff::Trainer trainer (nn_, detail::CrossEntropyWithLogitsLoss{});
            std::mt19937_64 rng(params_.random_state);
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(X.rows());

            for (i32 epoch = 0; epoch < params_.max_iter; ++epoch) {

                perm.setIdentity();
                std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rng);

                const MatrixR X_shuffled = perm * X;
                const MatrixC y_shuffled = perm * y_one_hot_mat;

                Scalar epoch_loss_sum = 0.0f;
                i32 total_samples = 0;

                for (i32 start = 0; start < X.rows(); start += params_.batch_size) {
                    const i32 end = std::min(start + params_.batch_size, static_cast<i32>(X.rows()));
                    const auto X_batch = X_shuffled.middleRows(start, end - start);
                    const MatrixC y_batch = y_shuffled.middleRows(start, end - start);

                    // zero grad
                    std::visit([&](auto&& optimizer) {
                        optimizer.zeroGrad(nn_);
                    }, opt);

                    const auto [d_loss, r_loss, t_loss] = trainer.trainStep(X_batch, y_batch);

                    // optimizer step
                    std::visit([&](auto&& optimizer) {
                        optimizer.step(nn_);
                    }, opt);

                    i32 current_batch_size = end - start;
                    epoch_loss_sum += t_loss * current_batch_size;
                    total_samples += current_batch_size;
                }

                const Scalar iteration_loss = epoch_loss_sum / total_samples;
                if (params_.callback_fn) {
                    params_.callback_fn(iteration_loss, epoch);
                }
                if (iteration_loss <= params_.tolerance) {
                    break;
                }
            }
            is_fitted_ = true;
        }

        [[nodiscard]] const LabelEncoderInternal & get_encoder_() const override {return encoder_;}

    private:
        MLPParams params_;
        LabelEncoderInternal encoder_;
        ff::FeedForwardNN nn_;
        int64_t n_features{};
        int64_t n_classes{};
        bool is_fitted_ = false;
    };

}

#endif //AXML_MLP_HPP