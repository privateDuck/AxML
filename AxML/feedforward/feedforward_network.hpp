#pragma once

#ifndef AXML_FEEDFORWARD_NETWORK_HPP
#define AXML_FEEDFORWARD_NETWORK_HPP

#include "feedforward_base.hpp"

namespace AxML::ff {

    struct Layer {
        MatrixC weights;
        Vector biases;

        // Gradients (computed during backward pass)
        MatrixC weight_grad;
        Vector bias_grad;

        // Caches for backpropagation
        MatrixR z_cache;       // Pre-activation
        MatrixR a_cache;       // Post-activation
        MatrixR input_cache;   // Input to this layer

        detail::ActivationFunc activation;
        detail::RegularizationFunc regularization;

        Layer(const int in_size, const int out_size, const detail::ActivationFunc act, const detail::RegularizationFunc &reg = detail::NoRegularization{})
            : activation(act), regularization(reg) {
            // Xavier/Glorot initialization
            const Scalar limit = std::sqrt(6.0f / static_cast<Scalar>(in_size + out_size));
            weights = MatrixR::Random(in_size, out_size) * limit;
            biases = Vector::Zero(out_size);
        }

        template<typename Derived>
        MatrixR forward(const Eigen::MatrixBase<Derived>& input) {
            AXML_MATR_ASSERT(Derived)
            input_cache = input;
            z_cache = (input * weights).rowwise() + biases.transpose();

            const detail::ActivationResult act_result = std::visit(
                [&](const auto& act) { return act.forward(z_cache); },
                activation
            );

            a_cache = act_result.output;
            return a_cache;
        }

        template<typename Derived>
        MatrixR backward(const Eigen::MatrixBase<Derived>& grad_output) {
            AXML_MATR_ASSERT(Derived)
            // Compute gradient w.r.t. pre-activation
            MatrixR grad_z = std::visit(
                [&](const auto& act) { return act.backward(z_cache, grad_output); },
                activation
            );

            // Compute gradients for weights and biases
            weight_grad = input_cache.transpose() * grad_z;
            bias_grad = grad_z.colwise().sum();

            // Add regularization gradient to weight gradients
            detail::RegularizationResult reg_result = std::visit(
                [&](const auto& reg) { return reg.compute(weights); },
                regularization
            );
            weight_grad += reg_result.weight_gradient;

            // Compute gradient w.r.t. input (for previous layer)
            return grad_z * weights.transpose();
        }

        // Compute regularization penalty for this layer
        [[nodiscard]] Scalar getRegularizationPenalty() const {
            return std::visit(
                [&](const auto& reg) { return reg.compute(weights).penalty; },
                regularization
            );
        }
    };

    class FeedForwardNN {
    public:
        FeedForwardNN() = default;

        void addLayer(int in_size, int out_size, detail::ActivationFunc activation,
                      detail::RegularizationFunc regularization = detail::NoRegularization{}) {
            layers.emplace_back(in_size, out_size, std::move(activation), std::move(regularization));
        }

        [[nodiscard]] MatrixR forward(const MatrixR& input) {
            MatrixR curr = input;
            for (auto& layer : layers) {
                curr = layer.forward(curr);
            }
            return curr;
        }

        void backward(const MatrixR& grad_output) {
            MatrixR grad = grad_output;
            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
                grad = layers[i].backward(grad);
            }
        }

        // Compute total regularization penalty across all layers
        [[nodiscard]] Scalar getRegularizationPenalty() const {
            Scalar total = 0.0f;
            for (const auto& layer : layers) {
                total += layer.getRegularizationPenalty();
            }
            return total;
        }

        [[nodiscard]] std::vector<Layer>& getLayers() { return layers; }
        [[nodiscard]] const std::vector<Layer>& getLayers() const { return layers; }

        // Zero out all gradients
        void zeroGrad() {
            for (auto& layer : layers) {
                layer.weight_grad.setZero();
                layer.bias_grad.setZero();
            }
        }

    private:
        std::vector<Layer> layers;
    };

    class Trainer {
    public:
        Trainer(FeedForwardNN& model, detail::LossFunc loss_fn)
            : model_(model), loss_fn_(std::move(loss_fn)) {}

        // Single training step
        // Returns: {data_loss, regularization_penalty, total_loss}
        template <typename DerivedX, typename DerivedY>
        [[nodiscard]] std::tuple<Scalar, Scalar, Scalar> trainStep(const Eigen::MatrixBase<DerivedX>& input, const Eigen::MatrixBase<DerivedY>& targets) {
            AXML_MATR_ASSERT(DerivedX)
            AXML_FVEC_ASSERT(DerivedY)
            // Forward pass
            MatrixR predictions = model_.forward(input);

            // Compute loss and its gradient
            auto [value, gradient] = std::visit(
                [&](const auto& loss) { return loss.forward(predictions, targets); },
                loss_fn_
            );

            // Get regularization penalty
            Scalar reg_penalty = model_.getRegularizationPenalty();

            // Total loss = data loss + regularization penalty
            Scalar total_loss = value + reg_penalty;

            // Backward pass (regularization gradients are added automatically in Layer::backward)
            model_.backward(gradient);

            return {value, reg_penalty, total_loss};
        }

        // Evaluation (no gradient computation needed in model)
        // Returns: {data_loss, regularization_penalty, total_loss}
        [[nodiscard]] std::tuple<Scalar, Scalar, Scalar> evaluate(const MatrixR& input, const MatrixR& targets) {
            MatrixR predictions = model_.forward(input);

            auto [value, gradient] = std::visit(
                [&](const auto& loss) { return loss.forward(predictions, targets); },
                loss_fn_
            );

            Scalar reg_penalty = model_.getRegularizationPenalty();
            Scalar total_loss = value + reg_penalty;

            return {value, reg_penalty, total_loss};
        }

        // Get predictions without computing loss
        [[nodiscard]] MatrixR predict(const MatrixR& input) const {
            return model_.forward(input);
        }

    private:
        FeedForwardNN& model_;
        detail::LossFunc loss_fn_;
    };
}

#endif //AXML_FEEDFORWARD_NETWORK_HPP