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
        LinearModelProblem(const Scalar* X_ptr, const Scalar* y_ptr, const int64_t n_samples,
            const int64_t n_features, const int64_t n_outputs,
            LossFunc loss, RegularizationFunc reg)
            : X_ptr(X_ptr), y_ptr(y_ptr), n_samples_(n_samples), n_features_(n_features), n_outputs_(n_outputs), loss_fn_(std::move(loss)), reg_fn_(std::move(reg))
        {}

        // Compute loss
        Scalar operator()(const Vector& params, Vector& grad) {
            // unpack parameters into weights and biases
            const auto X_ = Eigen::Map<const MatrixC>(X_ptr, n_samples_, n_features_);
            const auto y_ = Eigen::Map<const MatrixC>(y_ptr, n_samples_, n_outputs_);
            auto W = Eigen::Map<const MatrixR>(
                params.data(), n_features_, n_outputs_
            );
            auto b = Eigen::Map<const Vector>(
                params.data() + n_features_ * n_outputs_, n_outputs_
            );

            // predictions = X * W + b
            MatrixC predictions = (X_ * W).rowwise() + b.transpose();

            // compute data loss and gradient
            auto [loss_value, loss_gradient] = std::visit(
                [&](const auto& loss) { return loss.forward(predictions, y_); },
                loss_fn_
            );

            // compute regularization
            auto [penalty, weight_gradient] = std::visit(
                [&](const auto& reg) { return reg.compute(W); },
                reg_fn_
            );

            const Scalar total_loss = loss_value + penalty;

            // compute gradients
            // dL/dW = X^T * loss_gradient + reg_gradient
            const MatrixR grad_W = X_.transpose() * loss_gradient + weight_gradient;

            // dL/db = sum(loss_gradient) across samples
            const Vector grad_b = loss_gradient.colwise().sum();

            // Pack gradients into single vector
            Eigen::Map<MatrixR>(grad.data(), n_features_, n_outputs_) = grad_W;
            Eigen::Map<Vector>(grad.data() + n_features_ * n_outputs_, n_outputs_) = grad_b;

            return total_loss;
        }

        [[nodiscard]] int64_t num_params() const {
            return n_features_ * n_outputs_ + n_outputs_;
        }

    private:
        const Scalar* X_ptr;
        const Scalar* y_ptr;
        int64_t n_samples_;
        int64_t n_features_;
        int64_t n_outputs_;
        LossFunc loss_fn_;
        RegularizationFunc reg_fn_;
    };

    // LINEAR MODEL BASE (uses LBFGS)
    /*class LinearModelLBFGS {
    public:
        LinearModelLBFGS(LossFunc loss, RegularizationFunc reg, const Scalar tolerance = 1e-6)
            : reg_fn_(std::move(reg)), loss_fn_(std::move(loss)), tol_(tolerance), fitted_(false) {}

        void fit(const ConstMatRRef& X, const Eigen::MatrixBase<DerivedY>& y, const int max_iterations = 100) {
            AXML_MATR_ASSERT(DerivedX)
            AXML_FVEC_ASSERT(DerivedY)
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

        template<typename Derived>
        [[nodiscard]] MatrixR predict(const Eigen::MatrixBase<Derived>& X) const {
            AXML_MATR_ASSERT(Derived)
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        template<typename DerivedX, typename DerivedY>
        [[nodiscard]] Scalar evaluate(const Eigen::MatrixBase<DerivedX>& X, const Eigen::MatrixBase<DerivedY>& y) const {
            AXML_MATR_ASSERT(DerivedX)
            AXML_FVEC_ASSERT(DerivedY)
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
    };*/

    // Simple Linear Regression (MSE Loss)
    /*class LinearRegression : public LinearModelLBFGS {
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
    };*/
}

#endif //AXML_LINEAR_BASE_HPP