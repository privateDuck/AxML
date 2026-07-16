#pragma once

#ifndef AXML_LINEAR_REGRESSION_HPP
#define AXML_LINEAR_REGRESSION_HPP

#include "base.hpp"
#include "feedforward/linear_base.hpp"

namespace AxML {

    class LinearRegression : public Regressor {
    public:
        explicit LinearRegression(
                const RegularizationType regularization = RegularizationType::NONE,
                const int64_t max_iter = 10000, const Scalar tolerance = 1e-6, const Scalar l1_reg = 0.01, const Scalar l2_reg = 0.01
            )
            : tol_(tolerance), max_iter_(max_iter), fitted_(false) {
            switch (regularization) {
                case RegularizationType::L1:
                    reg_fn_ = detail::L1Regularization{l1_reg};
                    break;
                case RegularizationType::L2:
                    reg_fn_ = detail::L2Regularization{l2_reg};
                    break;
                case RegularizationType::ElasticNet:
                    reg_fn_ = detail::ElasticNetRegularization{l1_reg, l2_reg};
                    break;
                case RegularizationType::NONE:
                default:
                    reg_fn_ = detail::NoRegularization{};
                    break;
            }
        }

        [[nodiscard]] Vector predict(const MatrixR &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        bool is_fitted() const override {
            return fitted_;
        }

        uint64_t dims() const override {
            return n_features_;
        }

        void reset() override {
            weights_.setZero();
            biases_.setZero();
            last_loss_ = 0.0;
            n_features_ = 0;
            n_outputs_ = 0;
            iterations_ = 0;
            fitted_ = false;
        }

        std::string name() const override {
            return "LinearRegression";
        }

        uint32_t type_id() const override {
            return ID_LINEAR_REGRESSION;
        }

        [[nodiscard]] const MatrixR& getWeights() const { return weights_; }
        [[nodiscard]] const Vector& getBiases() const { return biases_; }
        [[nodiscard]] Scalar getLastLoss() const { return last_loss_; }
        [[nodiscard]] int64_t getIterations() const { return iterations_; }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            if (!fitted_) {
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
                detail::LinearModelProblem problem(X, y, loss_fn_, reg_fn_);

                // Setup LBFGS
                LBFGSpp::LBFGSParam<Scalar> param;
                param.epsilon = tol_;
                param.max_iterations = max_iter_;

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
        }

    private:
        MatrixR weights_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        detail::LossFunc loss_fn_ = detail::MSELoss{};
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };

    // MSE Loss + L2 Regularization
    class RidgeRegression : public Regressor {
    public:
        explicit RidgeRegression(
                const Scalar alpha = 1.0, const int64_t max_iter = 10000, const Scalar tolerance = 1e-6
            )
            : reg_fn_(detail::L2Regularization{alpha}), tol_(tolerance), max_iter_(max_iter), fitted_(false) {
        }

        [[nodiscard]] Vector predict(const MatrixR &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        bool is_fitted() const override {
            return fitted_;
        }

        uint64_t dims() const override {
            return n_features_;
        }

        void reset() override {
            weights_.setZero();
            biases_.setZero();
            last_loss_ = 0.0;
            n_features_ = 0;
            n_outputs_ = 0;
            iterations_ = 0;
            fitted_ = false;
        }

        std::string name() const override {
            return "RidgeRegression";
        }

        uint32_t type_id() const override {
            return ID_RIDGE_REGRESSION;
        }

        [[nodiscard]] const MatrixR& getWeights() const { return weights_; }
        [[nodiscard]] const Vector& getBiases() const { return biases_; }
        [[nodiscard]] Scalar getLastLoss() const { return last_loss_; }
        [[nodiscard]] int64_t getIterations() const { return iterations_; }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            if (!fitted_) {
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
                detail::LinearModelProblem problem(X, y, loss_fn_, reg_fn_);

                // Setup LBFGS
                LBFGSpp::LBFGSParam<Scalar> param;
                param.epsilon = tol_;
                param.max_iterations = max_iter_;

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
        }

    private:
        MatrixR weights_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        detail::LossFunc loss_fn_ = detail::MSELoss{};
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };

    // MSE Loss + L1 Regularization
    class LassoRegression : public Regressor {
    public:
        explicit LassoRegression(
                const Scalar alpha = 1.0, const int64_t max_iter = 10000, const Scalar tolerance = 1e-6
            )
            : reg_fn_(detail::L1Regularization{alpha}), tol_(tolerance), max_iter_(max_iter), fitted_(false) {
        }

        [[nodiscard]] Vector predict(const MatrixR &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        bool is_fitted() const override {
            return fitted_;
        }

        uint64_t dims() const override {
            return n_features_;
        }

        void reset() override {
            weights_.setZero();
            biases_.setZero();
            last_loss_ = 0.0;
            n_features_ = 0;
            n_outputs_ = 0;
            iterations_ = 0;
            fitted_ = false;
        }

        std::string name() const override {
            return "LassoRegression";
        }

        uint32_t type_id() const override {
            return ID_LASSO_REGRESSION;
        }

        [[nodiscard]] const MatrixR& getWeights() const { return weights_; }
        [[nodiscard]] const Vector& getBiases() const { return biases_; }
        [[nodiscard]] Scalar getLastLoss() const { return last_loss_; }
        [[nodiscard]] int64_t getIterations() const { return iterations_; }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            if (!fitted_) {
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
                detail::LinearModelProblem problem(X, y, loss_fn_, reg_fn_);

                // Setup LBFGS
                LBFGSpp::LBFGSParam<Scalar> param;
                param.epsilon = tol_;
                param.max_iterations = max_iter_;

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
        }

    private:
        MatrixR weights_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        detail::LossFunc loss_fn_ = detail::MSELoss{};
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };

    // MSE Loss + L1 and L2 Regularization
    class ElasticNetRegression : public Regressor {
    public:
        explicit ElasticNetRegression(
                const Scalar l1_ratio = 0.5, const Scalar alpha = 1.0,
                const int64_t max_iter = 10000, const Scalar tolerance = 1e-6
            )
            : reg_fn_(detail::ElasticNetRegularization{l1_ratio * alpha, (1 - l1_ratio) * alpha}),
        tol_(tolerance), max_iter_(max_iter), fitted_(false) {
        }

        [[nodiscard]] Vector predict(const MatrixR &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        bool is_fitted() const override {
            return fitted_;
        }

        uint64_t dims() const override {
            return n_features_;
        }

        void reset() override {
            weights_.setZero();
            biases_.setZero();
            last_loss_ = 0.0;
            n_features_ = 0;
            n_outputs_ = 0;
            iterations_ = 0;
            fitted_ = false;
        }

        std::string name() const override {
            return "ElasticNetRegression";
        }

        uint32_t type_id() const override {
            return ID_ELASTICNET_REGRESSION;
        }

        [[nodiscard]] const MatrixR& getWeights() const { return weights_; }
        [[nodiscard]] const Vector& getBiases() const { return biases_; }
        [[nodiscard]] Scalar getLastLoss() const { return last_loss_; }
        [[nodiscard]] int64_t getIterations() const { return iterations_; }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            if (!fitted_) {
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
                detail::LinearModelProblem problem(X, y, loss_fn_, reg_fn_);

                // Setup LBFGS
                LBFGSpp::LBFGSParam<Scalar> param;
                param.epsilon = tol_;
                param.max_iterations = max_iter_;

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
        }

    private:
        MatrixR weights_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        detail::LossFunc loss_fn_ = detail::MSELoss{};
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };

}

#endif //AXML_LINEAR_REGRESSION_HPP