#pragma once

#ifndef AXML_SVC_HPP
#define AXML_SVC_HPP

#include "../base.hpp"
#include "../feedforward/linear_base.hpp"
#include "../rff.hpp"

namespace AxML {

    namespace detail {

        class MultiHingeProblem {
        public:
        MultiHingeProblem(const MatrixR& X, const Vector& y, RegularizationFunc reg)
            : X_(X), y_(y), loss_fn_(MultiHingeLoss{}), reg_fn_(std::move(reg)) {
            n_samples_ = X.rows();
            n_features_ = X.cols();
            n_outputs_ = y.cols();
        }

        // Compute loss
        Scalar operator()(const Vector& params, Vector& grad) {
            // unpack parameters into weights and biases
            auto W = Eigen::Map<const MatrixR>(
                params.data(), n_features_, n_outputs_
            );
            auto b = Eigen::Map<const Vector>(
                params.data() + n_features_ * n_outputs_, n_outputs_
            );

            // predictions = X * W + b
            MatrixR predictions = (X_ * W).rowwise() + b.transpose();

            // compute data loss and gradient
            auto [loss_value, loss_gradient] = loss_fn_.forward(predictions, y_);

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
        const MatrixR &X_;
        const Vector &y_;
        MultiHingeLoss loss_fn_;
        RegularizationFunc reg_fn_;
        int64_t n_samples_;
        int64_t n_features_;
        int64_t n_outputs_;
    };

    }

    class RBFSVC;

    class LinearSVC : public Classifier {
    public:
        explicit LinearSVC(const Scalar C = 1.0, const int64_t max_iter = 10000, const Scalar tolerance = 1e-6) :
        reg_fn_(detail::L2Regularization(2.0f / C)), tol_(tolerance),
        C_(C), max_iter_(max_iter), fitted_(false) {
        }

        VectorI predict(const ConstMatRRef &X) const override {
            if (X.cols() != n_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", n_features_, X.cols()));
            }
            MatrixR scores = lin_solve(X);
            VectorI labels(scores.rows());

            for (int i = 0; i < scores.rows(); ++i) {
                Eigen::Index max_idx;
                scores.row(i).maxCoeff(&max_idx);
                labels(i) = encoder_.inverse_transform(max_idx);
            }

            return labels;
        }

        [[nodiscard]] MatrixR decision_function(const MatrixR& X) const {
            return lin_solve(X);
        }

        bool supports_predict_proba() const noexcept override {return false;}

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override{return fitted_;}
        uint64_t dims() const override {return n_features_;}
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
            return "LinearSVC";
        }
        uint32_t type_id() const override {return ID_LINEAR_SV_CLASSIFIER;}

    protected:
        friend RBFSVC;

        const LabelEncoderInternal& get_encoder_() const override {
            return encoder_;
        }

        void fit_impl(const ConstMatRRef &X, const ConstVecIRef &y) override {
            // SVM convention: minimize (1/C)||w||² + Σ hinge_loss
            // Equivalent to: minimize hinge_loss + (1/2C)λ||w||²
            // So we need to adjust regularization strength based on C
            // λ = 2/C
            if (!fitted_) {
                n_features_ = X.cols();
                n_outputs_ = y.cols();

                const auto y_span = std::span(y.data(), y.size());
                encoder_.fit(y_span);
                const auto y_enc_scalar = encoder_.transform_to_float(y_span);
                const auto y_enc_vector = Eigen::Map<const Vector>(y_enc_scalar.data(), y_enc_scalar.size());

                // Initialize parameters (Xavier initialization)
                const Scalar limit = std::sqrt(6.0f / static_cast<Scalar>(n_features_ + n_outputs_));
                weights_ = MatrixR::Random(n_features_, n_outputs_) * limit;
                biases_ = Vector::Zero(n_outputs_);

                // Pack parameters into single vector for LBFGS
                Vector params(n_features_ * n_outputs_ + n_outputs_);
                Eigen::Map<MatrixR>(params.data(), n_features_, n_outputs_) = weights_;
                Eigen::Map<Vector>(params.data() + n_features_ * n_outputs_, n_outputs_) = biases_;

                // Create optimization problem
                // TODO: There is an inefficiency here. y_enc_vector is copied since the expected type is MatrixR
                // Fixed
                detail::MultiHingeProblem problem(X, y_enc_vector, reg_fn_);

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

        MatrixR lin_solve(const ConstMatRRef& X) const {
            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            return (X * weights_).rowwise() + biases_.transpose();
        }

        LabelEncoderInternal encoder_;
        MatrixR weights_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        Scalar tol_;
        Scalar C_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };

    class RBFSVC : public Classifier {
    public:
        explicit RBFSVC(
            const Scalar gamma = 1.0, const Scalar C = 1.0, const uint32_t rf_features = 100,
            const int64_t max_iter = 10000, const Scalar tolerance = 1e-6) :
        linear_svc_(C, max_iter, tolerance), gamma_(gamma), C_(C), rf_features_(rf_features) {}

        VectorI predict(const ConstMatRRef &X) const override {
            if (!linear_svc_.is_fitted()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR Z = rff_.transform(X);
            return linear_svc_.predict(Z);
        }

        [[nodiscard]] MatrixR decision_function(const ConstMatRRef& X) const {
            if (!linear_svc_.is_fitted()) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR Z = rff_.transform(X);
            return linear_svc_.decision_function(Z);
        }

        bool supports_predict_proba() const noexcept override {return false;}

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override{return linear_svc_.is_fitted();}
        uint64_t dims() const override {return linear_svc_.dims();}
        void reset() override {
            linear_svc_.reset();
            rff_.reset();
        }
        std::string name() const override {
            return "RBF_SVC";
        }
        uint32_t type_id() const override {return ID_RBF_SV_CLASSIFIER;}

    protected:
        const LabelEncoderInternal &get_encoder_() const override {
            return linear_svc_.encoder_;
        }
        void fit_impl(const ConstMatRRef &X, const ConstVecIRef &y) override {
            if (!linear_svc_.is_fitted()) {
                rff_.generate(X.cols(), rf_features_, gamma_);
                MatrixR Z = rff_.transform(X);
                linear_svc_.fit(Z, y);
            }
        }
    private:
        LinearSVC linear_svc_;
        RandomFourierFeatures rff_{};
        Scalar gamma_;
        Scalar C_;
        int32_t rf_features_;
    };
}

#endif //AXML_SVC_HPP