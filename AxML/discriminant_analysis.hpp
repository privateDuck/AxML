#pragma once
#ifndef AXML_DISCRIMINANT_ANALYSIS_HPP
#define AXML_DISCRIMINANT_ANALYSIS_HPP

#include "base.hpp"
#include "tensor3.hpp"
#include "label_encoder.hpp"

namespace AxML {

    class LinearDiscriminantAnalysis final : public Classifier {
    public:
        explicit LinearDiscriminantAnalysis() = default;

        void save(OutputArchive &ar) const override {}

        void load(InputArchive &ar) override {}

        [[nodiscard]] std::unique_ptr<Estimator> clone() const override { return std::make_unique<LinearDiscriminantAnalysis>(*this); }

        [[nodiscard]] bool is_fitted() const override { return is_fitted_; }

        [[nodiscard]] uint64_t dims() const override { return num_features_; }
        void reset() override {
            means_.setZero();
            log_priors_.setZero();
            precision_tensor_.setZero();
            log_det_cov_ = 0;
            is_fitted_ = false;
        }

        [[nodiscard]] std::string name() const override { return "LinearDiscriminantAnalysis"; }
        [[nodiscard]] uint32_t type_id() const override { return ID_LDA_CLASSIFIER; }

        [[nodiscard]] VectorI predict(const MatrixR &X) const override {
            if (X.cols() != num_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", num_features_, X.cols()));
            }
            if (!is_fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            VectorI preds(X.rows());
            MatrixR proba = predict_proba(X);
            for (i32 i = 0; i < X.rows(); ++i) {
                Eigen::Index arg;
                proba.row(i).maxCoeff(&arg);
                preds(i) = encoder_.inverse_transform(static_cast<i32>(arg));
            }
            return preds;
        }

        [[nodiscard]] bool supports_predict_proba() const noexcept override { return true; }

        [[nodiscard]] MatrixR predict_proba(const MatrixR &X) const override {
            if (X.cols() != num_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", num_features_, X.cols()));
            }
            if (!is_fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR logits = predict_log_proba(X);
            softmax_inplace(logits);
            return logits;
        }

    protected:
        [[nodiscard]] const LabelEncoderInternal &get_encoder_() const override {
            return encoder_;
        }

        void fit_impl(const MatrixR &X, const VectorI &y) override {
            const std::span<const i32> y_span(y.data(), y.size());
            encoder_.fit(y_span);
            const auto y_enc = encoder_.transform(y_span);

            num_classes_ = encoder_.num_unique_labels();
            num_features_ = static_cast<i32>(X.cols());

            const i32 N = static_cast<i32>(X.rows());
            means_.resize(num_classes_, num_features_);
            log_priors_.resize(num_classes_);

            MatrixC shared_cov = MatrixR::Zero(num_features_, num_features_);

            std::vector<i32> indices(N);
            i32 N_c = 0;
            for (i32 c = 0; c < num_classes_; ++c) {

                for (i32 i = 0; i < N; ++i) {
                    if (y_enc[i] == c) {
                        indices.at(N_c) = i;
                        ++N_c;
                    }
                }

                MatrixR Xc(N_c, num_features_);
                for (int i = 0; i < N_c; ++i) {
                    Xc.row(i) = X.row(indices[i]);
                }
                log_priors_(c) = std::log(static_cast<Scalar>(N_c) / N);
                const Vector mean = Xc.colwise().mean();
                means_.row(c) = mean;

                MatrixR centered = Xc.rowwise() - mean.transpose();
                shared_cov += centered.transpose() * centered;
                N_c = 0;
            }

            shared_cov /= (N - num_classes_);
            shared_cov += 1e-6 * MatrixC::Identity(num_features_, num_features_);

            // Cholesky
            const Eigen::LLT<MatrixC> llt(shared_cov);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("LDA SOLVE: Cholesky decomposition failed. The covariance matrix might not be positive definite.");
            }
            MatrixC L = llt.matrixL();

            // Precision matrix (Inverse covariance)
            precision_tensor_ = llt.solve(MatrixC::Identity(num_features_, num_features_));

            // log(det(cov)) = 2 * sum(log(diag(L)))
            log_det_cov_ = 2.0 * L.diagonal().array().log().sum();

            is_fitted_ = true;
        }

    private:

        [[nodiscard]] MatrixR predict_log_proba(const MatrixR &X) const {
            const auto dim = static_cast<Scalar>(num_features_);
            constexpr Scalar log_2pi = 1.8378770664093454835606594728112; // ln(2 * pi)
            const Scalar constant_term = -0.5f * (log_det_cov_ + dim * log_2pi);

            MatrixR log_proba(X.rows(), num_classes_);
            MatrixR diff(X.rows(), num_features_);

            for (i32 i = 0; i < num_classes_; ++i) {
                diff.noalias() = X.rowwise() - means_.row(i);
                auto mahalanobis = (diff * precision_tensor_).cwiseProduct(diff).rowwise().sum();
                log_proba.col(i).array() = (-0.5 * mahalanobis).array() + log_priors_(i) + constant_term;
            }
            return log_proba;
            /*
            // Term A: x^T * Sigma^-1 * x -> (batch, 1)
            MatrixC X_P = X.transpose() * precision_tensor_;
            const Vector term_x = (X_P.array() * X.array()).rowwise().sum();

            // Term B: mu^T * Sigma^-1 * mu -> (C)
            MatrixC Means_P = means_.transpose() * precision_tensor_;
            Vector term_mu = (Means_P.array() * means_.array()).rowwise().sum();

            // Term C: -2 * x^T * Sigma^-1 * mu -> (batch, C)
            const MatrixC term_interaction = -2.0f * (X_P * means_.transpose());

            // Mahalanobis Distance
            MatrixC mahalanobis_sq = term_x.replicate(1, num_classes_);
            for (int c = 0; c < num_classes_; ++c) {
                mahalanobis_sq.col(c).array() += term_mu(c);
            }
            mahalanobis_sq += term_interaction;

            MatrixR log_likelihood = constant_term - static_cast<Scalar>(0.5) * mahalanobis_sq.array();

            // Add priors
            for (int c = 0; c < num_classes_; ++c) {
                log_likelihood.col(c).array() += priors_(c);
            }

            return log_likelihood;
            */
        }

        static void softmax_inplace(MatrixR& logits) {
            for (i32 row = 0; row < logits.rows(); ++row) {
                auto max_val = logits.row(row).maxCoeff();
                Vector exp_logits = (logits.row(row).array() - max_val).exp();
                logits.row(row) = exp_logits /= exp_logits.sum();
            }
        }

        LabelEncoderInternal encoder_;
        MatrixC means_;               // (C, F)
        Vector log_priors_;               // (C)
        MatrixC precision_tensor_;    // (F, F)
        Scalar log_det_cov_{};        // (1)
        i32 num_classes_{};
        i32 num_features_{};
        bool is_fitted_ = false;
    };

    class QuadraticDiscriminantAnalysis final : public Classifier {
    public:
        explicit QuadraticDiscriminantAnalysis(const bool is_naive_bayes = false) : is_naive_bayes_(is_naive_bayes) {}

        bool supports_predict_proba() const noexcept override { return true; }
        void save(OutputArchive &ar) const override{}

        void load(InputArchive &ar) override{}

        std::unique_ptr<Estimator> clone() const override{return std::make_unique<QuadraticDiscriminantAnalysis>(*this);}

        void reset() override {
            means.setZero();
            priors.setZero();
            nb_variances.setZero();
            is_fitted_ = false;
        }

        [[nodiscard]] std::string name() const override { return "QuadraticDiscriminantAnalysis"; }

        [[nodiscard]] uint32_t type_id() const override { return ID_QDA_CLASSIFIER; }

        [[nodiscard]] bool is_fitted() const override { return is_fitted_; }

        [[nodiscard]] uint64_t dims() const override { return num_features_; }

        [[nodiscard]] VectorI predict(const MatrixR &X) const override {
            if (X.cols() != num_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", num_features_, X.cols()));
            }
            if (!is_fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            VectorI preds(X.rows());
            MatrixR proba = predict_proba(X);
            for (i32 i = 0; i < X.rows(); ++i) {
                Eigen::Index arg;
                proba.row(i).maxCoeff(&arg);
                preds(i) = encoder_.inverse_transform(static_cast<i32>(arg));
            }
            return preds;
        }

        [[nodiscard]] MatrixR predict_proba(const MatrixR &X) const override {
            if (X.cols() != num_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", num_features_, X.cols()));
            }
            if (!is_fitted_) {
                throw std::runtime_error("Model not fitted yet!");
            }
            MatrixR logits = predict_log_proba(X);
            softmax_inplace(logits);
            return logits;
        }

    protected:
        [[nodiscard]] const LabelEncoderInternal &get_encoder_() const override {
            return encoder_;
        }

        void fit_impl(const MatrixR &X, const VectorI &y) override {
            const std::span y_span(y.data(), y.size());
            encoder_.fit(y_span);
            const auto y_enc = encoder_.transform(y_span);

            num_classes_ = encoder_.num_unique_labels();
            num_features_ = static_cast<i32>(X.cols());

            const i32 N = X.rows();

            means.resize(num_classes_, num_features_);
            priors.resize(num_classes_);
            log_det_covs.resize(num_classes_);

            precision_tensors = Tensor3(num_classes_, num_features_, num_features_);

            std::vector<i32> indices(N);
            i32 N_c = 0;
            for (i32 c = 0; c < num_classes_; ++c) {
                for (i32 i = 0; i < N; ++i) {
                    indices.at(N_c) = i;
                    N_c += y_enc[i] == c ? 1 : 0;
                }

                MatrixR Xc(N_c, num_features_);
                for (int i = 0; i < N_c; ++i) {
                    Xc.row(i) = X.row(indices[i]);
                }

                priors(c) = std::log(static_cast<float>(N_c) / N);
                const Vector mean = Xc.colwise().mean();
                means.row(c) = mean;

                MatrixR centered = Xc.rowwise() - mean.transpose();

                if (!is_naive_bayes_)
                {
                    // Standard QDA: Full covariance matrix per class
                    MatrixC covariance = (centered.transpose() * centered) / (N_c - 1.0);
                    covariance += 1e-6 * MatrixC::Identity(num_features_, num_features_);

                    Eigen::LLT<MatrixC> llt(covariance);
                    MatrixC L = llt.matrixL();
                    const auto solved = llt.solve(MatrixC::Identity(num_features_, num_features_)).eval();
                    precision_tensors.set_dim1(c, solved);
                    log_det_covs(c) = 2.0 * L.diagonal().array().log().sum();
                }else {
                    // Naive Bayes: Diagonal covariance matrix (just variances)
                    Vector var = (centered.array().square().colwise().sum()) / (N_c - 1.0f);
                    var.array() += 1e-6;
                    nb_variances.row(c) = var;

                    // log_det for a diagonal matrix is just sum(log(var))
                    log_det_covs(c) = var.array().log().sum();
                }
            }

            is_fitted_ = true;
        }

        MatrixR predict_log_proba(const MatrixR &X) const {
            const i32 rows = X.rows();
            const Scalar dim = static_cast<Scalar>(num_features_);

            MatrixR log_likelihood = MatrixR::Zero(rows, num_classes_);

            for (i32 c = 0; c < num_classes_; ++c) {
                constexpr Scalar log_2pi = 1.8378770664093454835606594728112;
                // delta = X - mu_c -> (batch, F)
                MatrixR delta = X.rowwise() - means.row(c);
                Vector mahalanobis(rows);

                if (!is_naive_bayes_) {
                    // QDA: delta * Sigma^-1 * delta^T
                    Vector var = nb_variances.row(c);
                    mahalanobis = (delta.array().square() / var.transpose().array()).rowwise().sum();
                }else {
                    // Naive Bayes: sum( delta^2 / var )
                    Vector var = nb_variances.row(c);
                    mahalanobis = (delta.array().square() / var.transpose().array()).rowwise().sum();
                }
                Scalar constant_c = -0.5 * (log_det_covs(c) + dim * log_2pi);
                log_likelihood.col(c) = constant_c - 0.5 * mahalanobis.array();
                log_likelihood.col(c).array() += priors(c);
            }

            return log_likelihood;
        }

        static void softmax_inplace(MatrixR& logits) {
            for (i32 row = 0; row < logits.rows(); ++row) {
                auto max_val = logits.row(row).maxCoeff();
                Vector exp_logits = (logits.row(row).array() - max_val).exp();
                logits.row(row) = exp_logits /= exp_logits.sum();
            }
        }

    private:
        LabelEncoderInternal encoder_;
        MatrixC means;                      // (C, F)
        Vector priors;                      // (C)
        Tensor3 precision_tensors;          // (C, F, F) for QDA
        MatrixC nb_variances;               // (C, F) for Naive Bayes
        Vector log_det_covs;                // (C)
        i32 num_classes_{};
        i32 num_features_{};
        bool is_fitted_ = false;
        bool is_naive_bayes_ = false;
    };

}

#endif //AXML_DISCRIMINANT_ANALYSIS_HPP