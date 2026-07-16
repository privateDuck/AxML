#pragma once
#ifndef AXML_LINEAR_BASE_HPP
#define AXML_LINEAR_BASE_HPP

#include "feedforward_base.hpp"
#include "LBFGSpp/LBFGS.h"

namespace AxML {

    enum class RegularizationType {
        NONE = 0,
        L1 = 1,
        L2 = 2,
        ElasticNet = 3,
    };
}

namespace AxML::detail {

    class LinearModelProblem {
    public:
        LinearModelProblem(const MatrixR& X, const MatrixR& y, LossFunc loss, RegularizationFunc reg)
            : X_(X), y_(y), loss_fn_(std::move(loss)), reg_fn_(std::move(reg)) {
            n_samples_ = X.rows();
            n_features_ = X.cols();
            n_outputs_ = y.cols();
        }

        // LBFGSpp interface: compute loss
        Scalar operator()(const Vector& params, Vector& grad) {
            // Unpack parameters into weights and biases
            MatrixR W = Eigen::Map<const MatrixR>(
                params.data(), n_features_, n_outputs_
            );
            Vector b = Eigen::Map<const Vector>(
                params.data() + n_features_ * n_outputs_, n_outputs_
            );

            // Forward pass: predictions = X * W + b
            MatrixR predictions = (X_ * W).rowwise() + b.transpose();

            // Compute data loss and gradient
                auto [loss_value, loss_gradient] = std::visit(
                [&](const auto& loss) { return loss.forward(predictions, y_); },
                loss_fn_
            );

            // Compute regularization
            auto [penalty, weight_gradient] = std::visit(
                [&](const auto& reg) { return reg.compute(W); },
                reg_fn_
            );

            const Scalar total_loss = loss_value + penalty;

            // Compute gradients
            // dL/dW = X^T * loss_gradient + reg_gradient
            MatrixR grad_W = X_.transpose() * loss_gradient + weight_gradient;

            // dL/db = sum(loss_gradient) across samples
            Vector grad_b = loss_gradient.colwise().sum();

            // Pack gradients into single vector
            Eigen::Map<MatrixR>(grad.data(), n_features_, n_outputs_) = grad_W;
            Eigen::Map<Vector>(grad.data() + n_features_ * n_outputs_, n_outputs_) = grad_b;

            return total_loss;
        }

        [[nodiscard]] int64_t num_params() const {
            return n_features_ * n_outputs_ + n_outputs_;
        }

    private:
        const MatrixR& X_;
        const MatrixR& y_;
        LossFunc loss_fn_;
        RegularizationFunc reg_fn_;
        int64_t n_samples_;
        int64_t n_features_;
        int64_t n_outputs_;
    };

    // LINEAR MODEL BASE (uses LBFGS)
    class LinearModelLBFGS {
    public:
        LinearModelLBFGS(LossFunc loss, RegularizationFunc reg, const Scalar tolerance = 1e-6)
            : reg_fn_(std::move(reg)), loss_fn_(std::move(loss)), tol_(tolerance), fitted_(false) {}

        void fit(const MatrixR& X, const MatrixR& y, const int max_iterations = 100) {
            n_features_ = X.cols();
            n_outputs_ = y.cols();

            // Initialize parameters (Xavier initialization)
            const Scalar limit = std::sqrt(6.0f / static_cast<Scalar>(n_features_ + n_outputs_));
            weights_ = MatrixR::Random(n_features_, n_outputs_) * limit;
            biases_ = Vector::Zero(n_outputs_);

            // Pack parameters into single vector for LBFGS
            Vector params(n_features_ * n_outputs_ + n_outputs_);
            Eigen::Map<MatrixR>(params.data(), n_features_, n_outputs_) = weights_;
            Eigen::Map<Vector>(params.data() + n_features_ * n_outputs_, n_outputs_) = biases_;

            // Create optimization problem
            LinearModelProblem problem(X, y, loss_fn_, reg_fn_);

            // Setup LBFGS
            LBFGSpp::LBFGSParam<Scalar> param;
            param.epsilon = tol_;
            param.max_iterations = max_iterations;

            LBFGSpp::LBFGSSolver<Scalar> solver(param);
            Scalar final_loss;

            // Optimize
            const int niter = solver.minimize(problem, params, final_loss);

            // Unpack optimized parameters
            weights_ = Eigen::Map<MatrixR>(params.data(), n_features_, n_outputs_);
            biases_ = Eigen::Map<Vector>(params.data() + n_features_ * n_outputs_, n_outputs_);

            fitted_ = true;
            last_loss_ = final_loss;
            iterations_ = niter;
        }

        [[nodiscard]] MatrixR predict(const MatrixR& X) const {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        [[nodiscard]] Scalar evaluate(const MatrixR& X, const MatrixR& y) const {
            MatrixR predictions = predict(X);
            const auto value = std::visit(
                [&](const auto& loss) { return loss.forward_wo_grad(predictions, y); },
                loss_fn_
            );

            const auto penalty = std::visit(
                [&](const auto& reg) { return reg.compute_wo_grad(weights_); },
                reg_fn_
            );

            return value + penalty;
        }

        [[nodiscard]] const MatrixR& getWeights() const { return weights_; }
        [[nodiscard]] const Vector& getBiases() const { return biases_; }
        [[nodiscard]] Scalar getLastLoss() const { return last_loss_; }
        [[nodiscard]] int64_t getIterations() const { return iterations_; }

    protected:
        MatrixR weights_;
        RegularizationFunc reg_fn_;
        Vector biases_;
        LossFunc loss_fn_;
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        bool fitted_;
    };

    // Simple Linear Regression (MSE Loss)
    class LinearRegression : public LinearModelLBFGS {
    public:
        explicit LinearRegression(RegularizationFunc reg = NoRegularization{})
            : LinearModelLBFGS(MSELoss{}, std::move(reg)) {}
    };

    // Ridge Regression (MSE Loss + L2 Regularization)
    class RidgeRegression : public LinearModelLBFGS {
    public:
        explicit RidgeRegression(Scalar alpha = 1.0f)
            : LinearModelLBFGS(MSELoss{}, L2Regularization{alpha}) {}
    };

    // Lasso Regression (MSE Loss + L1 Regularization)
    class LassoRegression : public LinearModelLBFGS {
    public:
        explicit LassoRegression(Scalar alpha = 1.0f)
            : LinearModelLBFGS(MSELoss{}, L1Regularization{alpha}) {}
    };

    // ElasticNet Regression (MSE Loss + L1 + L2)
    class ElasticNetRegression : public LinearModelLBFGS {
    public:
        explicit ElasticNetRegression(Scalar l1_ratio = 0.5f, Scalar alpha = 1.0f)
            : LinearModelLBFGS(MSELoss{}, ElasticNetRegularization{l1_ratio * alpha, (1 - l1_ratio) * alpha}) {}
    };
}

#endif //AXML_LINEAR_BASE_HPP