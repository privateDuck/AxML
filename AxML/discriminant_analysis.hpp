#pragma once
#ifndef AXML_DISCRIMINANT_ANALYSIS_HPP
#define AXML_DISCRIMINANT_ANALYSIS_HPP

#include "base.hpp"

namespace AxML {

    class LinearDiscriminantAnalysis final : public Classifier {
    public:
        explicit LinearDiscriminantAnalysis() {}

        void save(OutputArchive &ar) const override;

        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override;

        uint64_t dims() const override { return num_features_; }
        void reset() override;

        std::string name() const override { return "LinearDiscriminantAnalysis"; }
        uint32_t type_id() const override { return ID_LDA_CLASSIFIER; }

        Vector predict(const MatrixR &X) const override {
            Vector preds(X.rows());
            MatrixR proba = predict_proba(X);
            for (i32 i = 0; i < X.rows(); ++i) {
                Eigen::Index arg;
                proba.row(i).maxCoeff(&arg);
                preds(i) = static_cast<Scalar>(classes_[arg]);
            }
            return preds;
        }

        bool supports_predict_proba() const noexcept override { return true; }

        MatrixR predict_proba(const MatrixR &X) const override {
            MatrixR logits = predict_log_proba(X);
            softmax_inplace(logits);
            return logits;
        }

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override {
            std::vector<i32> unique_labels(y.data(), y.data() + y.size());
            std::ranges::sort(unique_labels);
            unique_labels.erase(std::ranges::unique(unique_labels).begin(), unique_labels.end());
            classes_ = std::move(unique_labels);

            std::vector<i32> y_enc(static_cast<size_t>(y.size()));
            for (i32 i = 0; i < static_cast<i32>(y.size()); ++i) {
                auto it = std::ranges::lower_bound(unique_labels, y(i));
                y_enc[static_cast<size_t>(i)] = static_cast<i32>(it - unique_labels.begin());
            }

            num_classes_ = static_cast<i32>(unique_labels.size());
            num_features_ = static_cast<i32>(X.cols());

            i32 N = static_cast<i32>(X.rows());
            means_.resize(num_classes_, num_features_);
            priors_.resize(num_classes_);

            MatrixC shared_cov(num_features_, num_features_);

            std::vector<i32> indices(N);
            i32 N_c = 0;
            for (const auto class_label : classes_) {

                for (i32 i = 0; i < N; ++i) {
                    indices.at(N_c) = i;
                    N_c += y_enc[i] == class_label ? 1 : 0;
                }

                MatrixR Xc(N_c, num_features_);
                for (int i = 0; i < N_c; ++i) {
                    Xc.row(i) = X.row(indices[i]);
                }
                priors_(class_label) = std::log(static_cast<Scalar>(N_c) / N);
                const Vector mean = Xc.colwise().mean();
                means_.row(class_label) = mean;

                MatrixR centered = Xc.rowwise().mean() - mean.transpose();
                shared_cov += centered.transpose() * centered;
                N_c = 0;
            }

            shared_cov /= (N - num_classes_);
            shared_cov += 1e-6 * MatrixC::Identity(num_features_, num_features_);

            // Cholesky
            const Eigen::LLT<MatrixC> llt(shared_cov);
            MatrixC L = llt.matrixL();

            // Precision matrix (Inverse covariance)
            precision_tensor_ = llt.solve(MatrixC::Identity(num_features_, num_features_));

            // log(det(cov)) = 2 * sum(log(diag(L)))
            log_det_cov_ = 2.0f * L.diagonal().array().log().sum();
        }

    private:

        MatrixR predict_log_proba(const MatrixR &X) const {
            i32 batch = X.rows();
            const Scalar dim = static_cast<Scalar>(num_features_);
            constexpr Scalar log_2pi = 1.8378770664093454835606594728112; // ln(2 * pi)
            Scalar constant_term = -0.5f * (log_det_cov_ + dim * log_2pi);

            // Term A: x^T * Sigma^-1 * x -> (batch, 1)
            MatrixC X_P = X * precision_tensor_;
            const Vector term_x = (X_P.array() * X.array()).rowwise().sum();

            // Term B: mu^T * Sigma^-1 * mu -> (C)
            MatrixC Means_P = means_ * precision_tensor_;
            Vector term_mu = (Means_P.array() * means_.array()).rowwise().sum();

            // Term C: -2 * x^T * Sigma^-1 * mu -> (batch, C)
            const MatrixC term_interaction = -2.0f * (X_P * means_.transpose());

            // Mahalanobis Distance
            MatrixC mahalanobis_sq = term_x.replicate(1, num_classes_);
            for (int c = 0; c < num_classes_; ++c) {
                mahalanobis_sq.col(c).array() += term_mu(c);
            }
            mahalanobis_sq += term_interaction;

            MatrixR log_likelihood = Vector::Constant(num_classes_, constant_term) - static_cast<Scalar>(0.5) * mahalanobis_sq;

            // Add priors
            for (int c = 0; c < num_classes_; ++c) {
                log_likelihood.col(c).array() += priors_(c);
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

        MatrixC means_;               // (C, F)
        Vector priors_;               // (C)
        std::vector<i32> classes_;    // (C)
        MatrixC precision_tensor_;    // (F, F)
        Scalar log_det_cov_{};        // (1)
        i32 num_classes_{};
        i32 num_features_{};
    };

}

#endif //AXML_DISCRIMINANT_ANALYSIS_HPP