#pragma once
#ifndef AXML_PCA_HPP
#define AXML_PCA_HPP

#include "base.hpp"
#include "utils/randomized_svd.hpp"
#include "scaler.hpp"

namespace AxML {

    class PCA final : public Transformer {
    public:
        explicit PCA(const uint32_t n_components = 2, const bool use_full_svd = false)
            : n_components_(n_components), use_full_svd_(use_full_svd) {}

        void save(OutputArchive &ar) const override;

        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override {
            return fitted_;
        }

        uint64_t dims() const override {
            return n_components_;
        }

        void reset() override {
            fitted_ = false;
            scalar_.reset();
            projection_matrix_.setZero();
        }

        std::string name() const override;

        uint32_t type_id() const override;

        void fit(const ConstMatRRef &X) override {
            if (use_full_svd_) {
                const Eigen::JacobiSVD<MatrixC> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
                projection_matrix_ = svd.matrixV().leftCols(n_components_);
            }
            else {
                const utils::RandomizedSVD svd(n_components_);
                const auto [U, S, V_t] = svd.compute(X);
                projection_matrix_ = V_t.leftCols(n_components_);
            }
            scalar_.fit(X);
            fitted_ = true;
        }

        MatrixR transform(const ConstMatRRef &X) const override {
            if (!fitted_) {
                throw std::runtime_error("PCA not fitted yet!");
            }
            const MatrixR X_scaled = scalar_.transform(X);
            return X_scaled * projection_matrix_;
        }

        MatrixR fit_transform(const ConstMatRRef &X) override {
            fit(X);
            return transform(X);
        }

    private:
        MatrixC projection_matrix_;
        StandardScalar scalar_;
        uint32_t n_components_;
        bool use_full_svd_;
        bool fitted_ = false;
    };

}


#endif //AXML_PCA_HPP