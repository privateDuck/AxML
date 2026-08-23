#pragma once
#ifndef AXML_OPTIMIZER_HPP
#define AXML_OPTIMIZER_HPP

#include "feedforward_network.hpp"

namespace AxML::ff {
    class SGDOptimizer {
    public:
        explicit SGDOptimizer(const Scalar learning_rate = 0.01f, const Scalar momentum = 0.0f)
            : lr_(learning_rate), momentum_(momentum) {}

        void step(FeedForwardNN& model) {
            if (velocity_.empty()) {
                // Initialize velocity vectors
                for (const auto& layer : model.getLayers()) {
                    velocity_.emplace_back(
                        MatrixC::Zero(layer.weights.rows(), layer.weights.cols()),
                        Vector::Zero(layer.biases.size())
                    );
                }
            }

            auto& layers = model.getLayers();
            for (size_t i = 0; i < layers.size(); ++i) {
                // Update velocity
                velocity_[i].first = momentum_ * velocity_[i].first - lr_ * layers[i].weight_grad;
                velocity_[i].second = momentum_ * velocity_[i].second - lr_ * layers[i].bias_grad;

                // Update parameters
                layers[i].weights += velocity_[i].first;
                layers[i].biases += velocity_[i].second;
            }
        }

        static void zeroGrad(FeedForwardNN& model) {
            model.zeroGrad();
        }

    private:
        Scalar lr_;
        Scalar momentum_;
        std::vector<std::pair<MatrixC, Vector>> velocity_;
    };


    class AdamOptimizer {
    public:
        explicit AdamOptimizer(const Scalar learning_rate = 0.001f,
                              const Scalar beta1 = 0.9f,
                              const Scalar beta2 = 0.999f,
                              const Scalar epsilon = 1e-8f)
            : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

        void step(FeedForwardNN& model) {
            if (m_.empty()) {
                // Initialize moments
                for (const auto& layer : model.getLayers()) {
                    m_.emplace_back(
                        MatrixC::Zero(layer.weights.rows(), layer.weights.cols()),
                        Vector::Zero(layer.biases.size())
                    );
                    v_.emplace_back(
                        MatrixC::Zero(layer.weights.rows(), layer.weights.cols()),
                        Vector::Zero(layer.biases.size())
                    );
                }
            }

            t_++;
            auto& layers = model.getLayers();

            for (size_t i = 0; i < layers.size(); ++i) {
                // Update biased first moment estimate
                m_[i].first = beta1_ * m_[i].first + (1 - beta1_) * layers[i].weight_grad;
                m_[i].second = beta1_ * m_[i].second + (1 - beta1_) * layers[i].bias_grad;

                // Update biased second moment estimate
                v_[i].first = beta2_ * v_[i].first + (1 - beta2_) * layers[i].weight_grad.array().square().matrix();
                v_[i].second = beta2_ * v_[i].second + (1 - beta2_) * layers[i].bias_grad.array().square().matrix();

                // Compute bias-corrected moments
                MatrixC m_hat_w = m_[i].first / (1 - std::pow(beta1_, t_));
                Vector m_hat_b = m_[i].second / (1 - std::pow(beta1_, t_));
                MatrixC v_hat_w = v_[i].first / (1 - std::pow(beta2_, t_));
                Vector v_hat_b = v_[i].second / (1 - std::pow(beta2_, t_));

                // Update parameters
                layers[i].weights -= (lr_ * m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon_)).matrix();
                layers[i].biases -= (lr_ * m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon_)).matrix();
            }
        }

        static void zeroGrad(FeedForwardNN& model) {
            model.zeroGrad();
        }

    private:
        Scalar lr_;
        Scalar beta1_;
        Scalar beta2_;
        Scalar epsilon_;
        int t_;
        std::vector<std::pair<MatrixC, Vector>> m_;  // First moment
        std::vector<std::pair<MatrixC, Vector>> v_;  // Second moment
    };

    using OptimizerType = std::variant<SGDOptimizer, AdamOptimizer>;
}

#endif //AXML_OPTIMIZER_HPP