#pragma once

#ifndef AXML_FEEDFORWARD_BASE_HPP
#define AXML_FEEDFORWARD_BASE_HPP
#include <variant>
#include "AxML/common.hpp"


namespace AxML::detail {

    struct ActivationResult {
        MatrixR output;
        MatrixR z_cache;
    };

    struct ReLU {
        template<typename Derived>
        [[nodiscard]] ActivationResult forward(const Eigen::MatrixBase<Derived>& z) const {
            return {z.cwiseMax(0), z};
        }

        template<typename DerivedZ, typename DerivedGrad>
        [[nodiscard]] MatrixR backward(const Eigen::MatrixBase<DerivedZ>& z_cache, const Eigen::MatrixBase<DerivedGrad>& grad_output) const {
            return grad_output.cwiseProduct((z_cache.array() > 0).template cast<Scalar>().matrix());
        }
    };

    struct Sigmoid {
        template<typename Derived>
        [[nodiscard]] ActivationResult forward(const Eigen::MatrixBase<Derived>& z) const {
            MatrixR output = z.unaryExpr([](Scalar x) {
                return 1.0f / (1.0f + std::exp(-x));
            });
            return {output, z};
        }

        template<typename DerivedZ, typename DerivedGrad>
        [[nodiscard]] MatrixR backward(const Eigen::MatrixBase<DerivedZ>& z_cache, const Eigen::MatrixBase<DerivedGrad>& grad_output) const {
            // sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
            MatrixR a = z_cache.unaryExpr([](const Scalar x) {
                return 1.0f / (1.0f + std::exp(-x));
            });
            return grad_output.cwiseProduct((a.array() * (1 - a.array())).matrix());
        }
    };

    struct Tanh {
        template<typename Derived>
        [[nodiscard]] ActivationResult forward(const Eigen::MatrixBase<Derived>& z) const {
            MatrixR output = z.unaryExpr([](const Scalar x) { return std::tanh(x); });
            return {output, z};
        }

        template<typename DerivedZ, typename DerivedGrad>
        [[nodiscard]] MatrixR backward(const Eigen::MatrixBase<DerivedZ>& z_cache, const Eigen::MatrixBase<DerivedGrad>& grad_output) const {
            // tanh'(z) = 1 - tanh²(z)
            MatrixR a = z_cache.unaryExpr([](const Scalar x) { return std::tanh(x); });
            return grad_output.cwiseProduct((1 - a.array().square()).matrix());
        }
    };

    struct Softmax {
        template<typename Derived>
        [[nodiscard]] ActivationResult forward(const Eigen::MatrixBase<Derived>& z) const {
            MatrixR output = z;
            for (int i = 0; i < output.rows(); ++i) {
                Scalar maxVal = output.row(i).maxCoeff();
                output.row(i) = (output.row(i).array() - maxVal).exp();
                output.row(i) /= output.row(i).sum();
            }
            return {output, z};
        }

        // Softmax gradient is complex (Jacobian matrix per sample)
        // Usually combined with CrossEntropy for numerical stability
        // This is a simplified version
        template<typename DerivedZ, typename DerivedGrad>
        [[nodiscard]] MatrixR backward(const Eigen::MatrixBase<DerivedZ>& z_cache, const Eigen::MatrixBase<DerivedGrad>& grad_output) const {
            MatrixR a = forward(z_cache).output;
            MatrixR grad = MatrixR::Zero(grad_output.rows(), grad_output.cols());

            for (int i = 0; i < a.rows(); ++i) {
                // Jacobian: diag(a) - a * a^T
                Vector ai = a.row(i).transpose();
                MatrixC jacobian = ai.asDiagonal();
                jacobian.noalias() -= ai * ai.transpose();
                grad.row(i) = (jacobian * grad_output.row(i).transpose()).transpose();
            }
            return grad;
        }
    };

    struct Linear {
        template<typename Derived>
        ActivationResult forward(const Eigen::MatrixBase<Derived>& z) const {
            return {z, z};
        }

        template<typename DerivedZ, typename DerivedGrad>
        [[nodiscard]] MatrixR backward(const Eigen::MatrixBase<DerivedZ>& z_cache, const Eigen::MatrixBase<DerivedGrad>& grad_output) const {
            return grad_output;
        }
    };

    using ActivationFunc = std::variant<ReLU, Sigmoid, Tanh, Softmax, Linear>;

    struct RegularizationResult {
        Scalar penalty;           // Regularization penalty to add to loss
        MatrixC weight_gradient;  // Gradient to add to weight gradients
    };

    struct NoRegularization {
        template<typename Derived>
        RegularizationResult compute(const Eigen::MatrixBase<Derived>& weights) const {
            return {0.0, MatrixC::Zero(weights.rows(), weights.cols())};
        }

        template<typename Derived>
        Scalar compute_wo_grad(const Eigen::MatrixBase<Derived>& weights) const {
            return 0.0;
        }
    };

    struct L2Regularization {
        Scalar lambda;  // Regularization strength

        explicit L2Regularization(const Scalar l = 0.01) : lambda(l) {}

        template<typename Derived>
        RegularizationResult compute(const Eigen::MatrixBase<Derived>& weights) const {
            // L2 penalty: λ/2 * ||W||²
            const Scalar penalty = 0.5 * lambda * weights.array().square().sum();

            // L2 gradient: λ * W
            const MatrixC gradient = lambda * weights;

            return {penalty, gradient};
        }

        template<typename Derived>
        Scalar compute_wo_grad(const Eigen::MatrixBase<Derived>& weights) const {
            return 0.5 * lambda * weights.array().square().sum();
        }
    };

    struct L1Regularization {
        Scalar lambda;  // Regularization strength

        explicit L1Regularization(const Scalar l = 0.01) : lambda(l) {}

        template<typename Derived>
        RegularizationResult compute(const Eigen::MatrixBase<Derived>& weights) const {
            // L1 penalty: λ * ||W||
            const Scalar penalty = lambda * weights.array().abs().sum();

            // L1 gradient: λ * sign(W)
            const MatrixC gradient = lambda * weights.array().sign().matrix();

            return {penalty, gradient};
        }

        template<typename Derived>
        Scalar compute_wo_grad(const Eigen::MatrixBase<Derived>& weights) const {
            return lambda * weights.array().abs().sum();
        }
    };

    struct ElasticNetRegularization {
        Scalar l1_lambda;  // L1 regularization strength
        Scalar l2_lambda;  // L2 regularization strength

        explicit ElasticNetRegularization(const Scalar l1 = 0.01, const Scalar l2 = 0.01)
            : l1_lambda(l1), l2_lambda(l2) {}

        template<typename Derived>
        RegularizationResult compute(const Eigen::MatrixBase<Derived>& weights) const {
            // Elastic Net: a * L1 + (1 - a) * L2
            // We use explicit l1_lambda and l2_lambda for more control
            const Scalar l1_penalty = l1_lambda * weights.array().abs().sum();
            const Scalar l2_penalty = 0.5 * l2_lambda * weights.array().square().sum();

            const MatrixC l1_grad = l1_lambda * weights.array().sign().matrix();
            const MatrixC l2_grad = l2_lambda * weights;

            return {l1_penalty + l2_penalty, l1_grad + l2_grad};
        }

        template<typename Derived>
        Scalar compute_wo_grad(const Eigen::MatrixBase<Derived>& weights) const {
            const Scalar l1_penalty = l1_lambda * weights.array().abs().sum();
            const Scalar l2_penalty = 0.5 * l2_lambda * weights.array().square().sum();
            return l1_penalty + l2_penalty;
        }
    };

    using RegularizationFunc = std::variant<NoRegularization, L1Regularization, L2Regularization, ElasticNetRegularization>;

    struct LossResult {
        Scalar value;
        MatrixC gradient;  // Gradient w.r.t predictions
    };

    struct MSELoss {
        LossResult forward(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = predictions - targets;
            const Scalar n = static_cast<Scalar>(predictions.rows());
            const Scalar loss = diff.array().square().sum() / n;
            const MatrixC grad = (2.0 / n) * diff;
            return {loss, grad};
        }

        Scalar forward_wo_grad(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = predictions - targets;
            const Scalar n = static_cast<Scalar>(predictions.rows());
            const Scalar loss = diff.array().square().sum() / n;
            return loss;
        }
    };

    struct MAELoss {
        LossResult forward(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = predictions - targets;
            const Scalar n = static_cast<Scalar>(predictions.rows());
            const Scalar loss = diff.array().abs().sum() / n;
            const MatrixC grad = diff.array().sign().matrix() / n;
            return {loss, grad};
        }

        Scalar forward_wo_grad(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = predictions - targets;
            const Scalar n = static_cast<Scalar>(predictions.rows());
            return diff.array().abs().sum() / n;
        }
    };

    struct CrossEntropyLoss {
        // Works with Softmax output (probabilities)
        // Targets must be one-hot encoded
        LossResult forward(const MatrixR& predictions, const MatrixR& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            // Clip predictions for numerical stability
            const MatrixR safe_pred = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
            // Loss: -sum(y_true * log(y_pred)) / n
            const Scalar loss = -(targets.array() * safe_pred.array().log()).sum() / n;
            // Gradient: (y_pred - y_true) / n
            // This is the combined gradient of Softmax + CrossEntropy
            const MatrixR grad = (predictions - targets) / n;

            return {loss, grad};
        }

        Scalar forward_wo_grad(const MatrixR& predictions, const MatrixR& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            const MatrixR safe_pred = predictions.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
            const Scalar loss = -(targets.array() * safe_pred.array().log()).sum() / n;
            return loss;
        }
    };

    struct CrossEntropyWithLogitsLoss {
        // Works with linear output (logits)
        // Targets must be one-hot encoded
        static LossResult forward(const ConstMatCRef& logits, const ConstMatCRef& targets) {
            const auto n = static_cast<Scalar>(logits.rows());

            // Compute Softmax
            MatrixC shifted_logits = logits.colwise() - logits.rowwise().maxCoeff();
            MatrixC exp_preds = shifted_logits.array().exp();
            const MatrixC probabilities = exp_preds.array().colwise() / exp_preds.rowwise().sum().array();

            // Compute Loss
            const auto safe_pred = probabilities.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
            const Scalar loss = -(targets.array() * safe_pred.array().log()).sum() / n;

            // Combined Gradient w.r.t logits
            const MatrixC grad = (probabilities - targets) / n;

            return {loss, grad};
        }

        static Scalar forward_wo_grad(const ConstMatCRef& logits, const ConstMatCRef& targets) {
            const auto n = static_cast<Scalar>(logits.rows());

            MatrixC shifted_logits = logits.colwise() - logits.rowwise().maxCoeff();
            MatrixC exp_preds = shifted_logits.array().exp();
            MatrixC probabilities = exp_preds.array().colwise() / exp_preds.rowwise().sum().array();

            const auto safe_pred = probabilities.cwiseMax(1e-7f).cwiseMin(1.0f - 1e-7f);
            return -(targets.array() * safe_pred.array().log()).sum() / n;
        }
    };

    struct HuberLoss {
        Scalar delta = 1.0f;

        explicit HuberLoss(const Scalar d = 1.0f) : delta(d) {}

        LossResult forward(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = (predictions - targets).eval();
            const Scalar n = static_cast<Scalar>(predictions.rows());

            Scalar loss = 0;
            MatrixC grad = Vector::Zero(diff.rows());

            for (int i = 0; i < diff.size(); ++i) {
                const Scalar abs_err = std::abs(diff(i));
                if (abs_err <= delta) {
                    loss += 0.5f * abs_err * abs_err;
                    grad(i) = diff(i);
                } else {
                    loss += delta * (abs_err - 0.5f * delta);
                    grad(i) = delta * (diff(i) > 0 ? 1.0f : -1.0f);
                }
            }

            return {loss / n, grad / n};
        }

        Scalar forward_wo_grad(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const auto diff = (predictions - targets).eval();
            const Scalar n = static_cast<Scalar>(predictions.rows());
            Scalar loss = 0;
            for (int i = 0; i < diff.size(); ++i) {
                if (const Scalar abs_err = std::abs(diff(i)); abs_err <= delta) {
                    loss += 0.5 * abs_err * abs_err;
                } else {
                    loss += delta * (abs_err - 0.5 * delta);
                }
            }

            return loss / n;
        }
    };

    struct MultiHingeLoss {
        Scalar margin = 1.0f;

        explicit MultiHingeLoss(const Scalar m = 1.0f) : margin(m) {}

        LossResult forward(const ConstMatCRef& predictions, const ConstMatCRef& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            Scalar total_loss = 0;
            MatrixC grad = MatrixC::Zero(predictions.rows(), predictions.cols());

            for (int i = 0; i < n; ++i) {
                int correct_class;
                targets.row(i).maxCoeff(&correct_class);
                const Scalar correct_score = predictions(i, correct_class);
                int violation_count = 0;

                for (int j = 0; j < predictions.cols(); ++j) {
                    if (j == correct_class) continue;

                    const Scalar loss_ij = margin - correct_score + predictions(i, j);
                    if (loss_ij > 0) {
                        total_loss += loss_ij;
                        grad(i, j) = 1.0f;
                        violation_count++;
                    }
                }

                grad(i, correct_class) = -static_cast<Scalar>(violation_count);
            }

            return {total_loss / n, grad / n};
        }

        Scalar forward_wo_grad(const ConstMatRRef& predictions, const ConstVecRef& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            Scalar total_loss = 0;
            for (int i = 0; i < n; ++i) {
                int correct_class = static_cast<int>(targets(i));
                const Scalar correct_score = predictions(i, correct_class);
                int violation_count = 0;

                for (int j = 0; j < predictions.cols(); ++j) {
                    if (j == correct_class) continue;

                    const Scalar loss_ij = margin - correct_score + predictions(i, j);
                    if (loss_ij > 0) {
                        total_loss += loss_ij;
                        violation_count++;
                    }
                }
            }

            return total_loss / n;
        }
    };

    struct ZeroOneLoss {
        // Non-differentiable, for evaluation only
        LossResult forward(const ConstVecRef& predictions, const ConstVecRef& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            Scalar errors = 0;

            for (int i = 0; i < n; ++i) {
                int pred_idx, target_idx;
                predictions.row(i).maxCoeff(&pred_idx);
                targets.row(i).maxCoeff(&target_idx);
                if (pred_idx != target_idx) errors++;
            }

            // Zero gradient (not trainable)
            return {errors / n, Vector::Zero(predictions.rows())};
        }

        Scalar forward_wo_grad(const ConstVecRef& predictions, const ConstVecRef& targets) const {
            const Scalar n = static_cast<Scalar>(predictions.rows());
            Scalar errors = 0;

            for (int i = 0; i < n; ++i) {
                int pred_idx, target_idx;
                predictions.row(i).maxCoeff(&pred_idx);
                targets.row(i).maxCoeff(&target_idx);
                if (pred_idx != target_idx) errors++;
            }
            return errors / n;
        }
    };

    using LossFunc = std::variant<MSELoss, MAELoss, CrossEntropyLoss, CrossEntropyWithLogitsLoss, HuberLoss, MultiHingeLoss, ZeroOneLoss>; // Multi Hinge Loss not part of this anymore.
}

#endif //AXML_FEEDFORWARD_BASE_HPP