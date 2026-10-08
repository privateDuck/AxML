#pragma once
#ifndef AXML_LIBSVC_HPP
#define AXML_LIBSVC_HPP

#include "base.hpp"
#include "svm.h"

namespace AxML {

    class SVC final : public Classifier {
    public:
        explicit SVC(const SVCParams& params) : n_features_(0), n_classes_(0), is_fitted_(false) {
            parameters_.svm_type = C_SVC;
            parameters_.kernel_type = RBF;
            parameters_.degree = 3;
            parameters_.gamma = 0; // 1/num_features
            parameters_.coef0 = 0;
            parameters_.nu = 0.5;
            parameters_.cache_size = 100; // in MB
            parameters_.C = params.C;
            parameters_.eps = params.tolerance;
            parameters_.p = 0.1;
            parameters_.shrinking = 1;
            parameters_.probability = 1;
            parameters_.nr_weight = 0;
            parameters_.weight_label = nullptr;
            parameters_.weight = nullptr;
        }

        [[nodiscard]] bool supports_predict_proba() const noexcept override {return enable_proba_;}
        void save(OutputArchive &ar) const override {}

        void load(InputArchive &ar) override {}

        std::unique_ptr<Estimator> clone() const override { return std::make_unique<SVC>(*this); }

        bool is_fitted() const override { return is_fitted_; }

        uint64_t dims() const override { return n_features_;}

        void reset() override {
            if (model_) {
                svm_free_and_destroy_model(&model_);
            }
        }

        std::string name() const override { return "SVC"; }

        uint32_t type_id() const override { return ID_SVC; }

        [[nodiscard]] VectorI predict(const ConstMatRRef &X) const override {
            if (!is_fitted_) {
                throw std::runtime_error(std::string("Model is not fitted yet."));
            }
            if (X.cols() != n_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", n_features_, X.cols()));
            }
            const auto N = X.rows();
            VectorI preds(N);
            std::vector<svm_node> nodes(n_features_ + 1);
            for (i64 i = 0; i < N; i++) {
                for (i32 col = 0; col < n_features_; col++) {
                    nodes[col] = svm_node{col + 1, X(i, col)};
                }
                nodes[n_features_ + 1] = svm_node{-1, 0.0};
                preds(i) = encoder_.inverse_transform(svm_predict(model_, nodes.data()));
            }
            return preds;
        }

        [[nodiscard]] MatrixR predict_proba(const ConstMatRRef &X) const override {
            if (!is_fitted_) {
                throw std::runtime_error(std::string("Model is not fitted yet."));
            }
            if (!enable_proba_) {
                throw std::runtime_error(std::string("Probability estimates are not enabled for this model."));
            }
            if (X.cols() != n_features_) {
                throw std::runtime_error(std::format("Model was fitted with {} dimensions. X has {} dimensions", n_features_, X.cols()));
            }
            const auto N = X.rows();
            MatrixR probas(N, n_classes_);
            std::vector<svm_node> nodes(n_features_ + 1);
            for (i64 i = 0; i < N; i++) {
                for (i32 col = 0; col < n_features_; col++) {
                    nodes[col] = svm_node{col + 1, X(i, col)};
                }
                nodes[n_features_ + 1] = svm_node{-1, 0.0};
                svm_predict_probability(model_, nodes.data(), probas.data() + i * n_classes_);
            }
            return probas;
        }

        ~SVC() override {
            if (model_) {
                svm_free_and_destroy_model(&model_);
            }
        }

    protected:
        void fit_impl(const ConstMatRRef &X, const ConstVecIRef &y) override {
            // Construct LibSVM structures
            n_features_ = X.cols();
            const auto N = X.rows();
            encoder_.fit(y);
            n_classes_ = encoder_.num_unique_labels();
            auto y_enc = encoder_.transform_to_float(y);

            std::vector<svm_node> nodes((n_features_ + 1) * N);
            std::vector<svm_node *> nodes_vec(N);
            const size_t col_stride = n_features_ + 1;
            for (i64 i = 0; i < N; i++) {
                for (i32 col = 0; col < n_features_; col++) {
                    nodes[i * col_stride + col] = svm_node{col + 1, X(i, col)};
                }
                nodes[i * col_stride + n_features_] = svm_node{-1, 0.0};
                nodes_vec[i] = &nodes[i * col_stride];
            }

            svm_problem problem;
            problem.l = N;
            problem.y = y_enc.data();
            problem.x = nodes_vec.data();

            const char* error = svm_check_parameter(&problem, &parameters_);
            if (error) {
                throw std::runtime_error(std::string("SVM parameter error: ") + error);
            }
            model_ = svm_train(&problem, &parameters_);

            is_fitted_ = true;
        }

        [[nodiscard]] const LabelEncoderInternal & get_encoder_() const override;

    private:
        svm_model* model_ = nullptr;
        LabelEncoderInternal encoder_;
        svm_parameter parameters_;
        i32 n_features_;
        i32 n_classes_;
        bool is_fitted_ = false;
        bool enable_proba_ = false;
    };

}

#endif //AXML_LIBSVC_HPP