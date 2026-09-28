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
        void save(OutputArchive &ar) const override;

        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override;

        uint64_t dims() const override;

        void reset() override;

        std::string name() const override;

        uint32_t type_id() const override;

        VectorI predict(const ConstMatRRef &X) const override;

    protected:
        detail::RegularizationFunc get_reg_fn() const {
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

        detail::ActivationFunc get_act_fn() const {
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
            const auto y_enc = encoder_.transform_to_float(y);
            const auto act = get_act_fn();
            const auto reg = get_reg_fn();

            i32 in_dim = X.cols();
            i32 out_dim = 0;
            for (i32 i = 0; i < params_.hidden_layers.size(); ++i) {
                if (i < params_.hidden_layers.size() - 1) {
                    out_dim = params_.hidden_layers[i + 1];
                }
                else {
                    out_dim = encoder_.num_unique_labels();
                }
                nn_.addLayer(in_dim, out_dim, act, reg);
                in_dim = params_.hidden_layers[i];
            }

            ff::OptimizerType opt;
            if (params_.solver == "sgd") {
                opt = ff::SGDOptimizer(params_.learning_rate, params_.momentum);
            }
            else if (params_.solver == "adam") {
                opt = ff::AdamOptimizer(params_.learning_rate, params_.beta_1, params_.beta_2, params_.epsilon);
            }
            else {
                throw std::runtime_error("Solver '" + params_.solver + "' is not supported.");
            }

            ff::Trainer trainer (nn_, detail::CrossEntropyLoss{});

            for (i32 iter = 0; iter < params_.max_iter; ++iter) {
                for (i32 start = 0; start < X.rows(); start += params_.batch_size) {
                    const i32 end = std::min(start + params_.batch_size, static_cast<i32>(X.rows()));
                    const auto X_batch = X.middleRows(start, end - start);
                    const auto y_batch = MapConstVector(y_enc.data() + start, end - start);

                    // zero grad
                    std::visit([&](auto&& optimizer) {
                        optimizer.zeroGrad(nn_);
                    }, opt);

                    const auto [d_loss, r_loss, _] = trainer.trainStep(X_batch, y_batch);

                    // optimizer step
                    std::visit([&](auto&& optimizer) {
                        optimizer.step(nn_);
                    }, opt);

                    if (params_.callback_fn) {
                        params_.callback_fn(d_loss, r_loss, iter);
                    }
                }
            }
        }

        const LabelEncoderInternal & get_encoder_() const override;

    private:
        MLPParams params_;
        LabelEncoderInternal encoder_;
        ff::FeedForwardNN nn_;
    };

}

#endif //AXML_MLP_HPP