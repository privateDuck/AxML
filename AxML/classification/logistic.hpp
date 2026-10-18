#pragma once
#ifndef AXML_LOGISTIC_HPP
#define AXML_LOGISTIC_HPP

#include "../core/base.hpp"
#include "../feedforward/linear_base.hpp"
#include "../label_encoder.hpp"

namespace AxML {

    class LogisticRegression : public Classifier {
    public:
        explicit LogisticRegression(const LogisticRegressionParams& params = LogisticRegressionParams())
        : tol_(params.tolerance), max_iter_(params.max_iterations), fitted_(false) {
            if (params.regularization == "l1") {
                reg_fn_ = detail::L1Regularization{params.l1_regularization};
            }
            else if (params.regularization == "l2") {
                reg_fn_ = detail::L2Regularization{params.l2_regularization};
            }
            else if (params.regularization == "elastic_net") {
                reg_fn_ = detail::ElasticNetRegularization{params.l1_regularization, params.l2_regularization};
            }
            else {
                reg_fn_ = detail::NoRegularization{};
            }
        }

        VectorI predict(const ConstMatRRef& X) const override {
            if (X.cols() != n_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", n_features_, X.cols()));
            }

            const MatrixR logits = (X * weights_).rowwise() + biases_.transpose();
            VectorI labels(logits.rows());

            for (int i = 0; i < logits.rows(); ++i) {
                Eigen::Index max_idx;
                logits.row(i).maxCoeff(&max_idx);
                labels(i) = encoder_.inverse_transform(max_idx);
            }

            return labels;
        }
        bool supports_predict_proba() const noexcept override { return true; }
        MatrixR predict_proba(const ConstMatRRef& X) const override {
            if (X.cols() != n_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", n_features_, X.cols()));
            }

            if (!fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }

            const MatrixR logits = (X * weights_).rowwise() + biases_.transpose();

            // Apply softmax for probabilities
            constexpr detail::Softmax softmax;
            return softmax.forward(logits).output;
        }

        void save(OutputArchive& ar) const override {}
        void load(InputArchive& ar) override {}

        std::unique_ptr<Estimator> clone() const override {
            return std::make_unique<LogisticRegression>(*this);
        }

        bool is_fitted() const override {return fitted_;}
        uint64_t dims() const override{return n_features_;}
        void reset() override {
            weights_.setZero();
            biases_.setZero();
            last_loss_ = 0.0;
            n_features_ = 0;
            n_outputs_ = 0;
            iterations_ = 0;
            fitted_ = false;
        }

        std::string name() const override { return "LogisticRegression"; }
        uint32_t type_id() const override { return ID_LOGISTIC_REGRESSION; }

    protected:
        const LabelEncoderInternal& get_encoder_() const override {
            return encoder_;
        }

        void fit_impl(const ConstMatRRef& X, const ConstVecIRef& y) override {
            if (!fitted_) {
                n_features_ = X.cols();
                const std::span<const i32> span_y(y.data(), y.size());
                encoder_.fit(span_y);
                n_outputs_ = encoder_.num_unique_labels();
                const std::vector<Scalar> y_enc = encoder_.transform_to_onehot_col_major(span_y);

                // Initialize parameters (Xavier initialization)
                const Scalar limit = std::sqrt(6.0f / static_cast<Scalar>(n_features_ + n_outputs_));
                weights_ = MatrixR::Random(n_features_, n_outputs_) * limit;
                biases_ = Vector::Zero(n_outputs_);

                // Pack parameters into single vector for LBFGS
                Vector params(n_features_ * n_outputs_ + n_outputs_);
                Eigen::Map<MatrixR>(params.data(), n_features_, n_outputs_) = weights_;
                Eigen::Map<Vector>(params.data() + n_features_ * n_outputs_, n_outputs_) = biases_;

                // Create optimization problem
                const MatrixC Xc = X;
                detail::LinearModelProblem problem(Xc.data(), y_enc.data(), Xc.rows(), n_features_, n_outputs_, loss_fn_, reg_fn_);

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
        LabelEncoderInternal encoder_;
        detail::RegularizationFunc reg_fn_;
        Vector biases_;
        detail::LossFunc loss_fn_ = detail::CrossEntropyWithLogitsLoss{};
        Scalar tol_;
        Scalar last_loss_{};
        int64_t n_features_{};
        int64_t n_outputs_{};
        int64_t iterations_{};
        int64_t max_iter_;
        bool fitted_;
    };
}

#endif //AXML_LOGISTIC_HPP