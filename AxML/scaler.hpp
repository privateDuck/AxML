#pragma once
#ifndef AXML_SCALER_HPP
#define AXML_SCALER_HPP

#include "base.hpp"

namespace AxML {

    class StandardScalar final : public Transformer {
    public:
        void save(OutputArchive &ar) const override;

        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override;

        uint64_t dims() const override;

        void reset() override;

        std::string name() const override;

        uint32_t type_id() const override;

        void fit(const ConstMatRRef &X) override {
            if (X.rows() == 0) {
                throw std::invalid_argument("Input matrix X must have at least one row.");
            }
            mean_ = X.colwise().mean();
            const auto centered = X.rowwise() - mean_.transpose();
            std_ = (centered.array().square().colwise().sum() / static_cast<Scalar>(X.rows() - 1)).array().sqrt();
            fitted_ = true;

            /*std_.resize(X.cols());
            for (i32 i = 0; i < X.cols(); i++) {
                std_(i) = std::sqrt((X.col(i).array() - mean_.array()).square().mean());
            }*/
        }

        MatrixR transform(const ConstMatRRef &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Scaler not fitted yet!");
            }
            if (X.cols() != mean_.size()) {
                throw std::invalid_argument("Input matrix X must have the same number of columns as the fitted scaler.");
            }
            return (X.rowwise() - mean_.transpose()).array().rowwise() / std_.transpose().array();
        }

        MatrixR fit_transform(const ConstMatRRef &X) override {
            if (!fitted_) {
                fit(X);
            }
            return transform(X);
        }

        MatrixR inverse_transform(const ConstMatRRef &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Scaler not fitted yet!");
            }
            if (X.cols() != mean_.size()) {
                throw std::runtime_error("Input matrix X must have the same number of columns as the fitted scaler.");
            }

            // X' * std + mu
            return X.array().rowwise() * std_.transpose().array() + mean_.transpose().array();
        }

    private:
        Vector mean_;
        Vector std_;
        bool fitted_ = false;
    };


    class MinMaxScalar final : Transformer {
    public:
        void save(OutputArchive &ar) const override;

        void load(InputArchive &ar) override;

        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override;

        uint64_t dims() const override;

        void reset() override;

        std::string name() const override;

        uint32_t type_id() const override;

        void fit(const ConstMatRRef &X) override {
            if (X.rows() == 0) {
                throw std::invalid_argument("Input matrix X must have an empty row.");
            }
            mins_ = X.colwise().minCoeff();
            ranges_ = X.colwise().maxCoeff() - mins_;
        }

        MatrixR transform(const ConstMatRRef &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Scaler not fitted yet!");
            }
            return (X.rowwise() - mins_.transpose()).array() / ranges_.transpose().array();
        }

        MatrixR fit_transform(const ConstMatRRef &X) override {
            if (!fitted_) {
                fit(X);
            }
            return transform(X);
        }

        MatrixR inverse_transform(const ConstMatRRef &X) const override {
            if (!fitted_) {
                throw std::runtime_error("Scaler not fitted yet!");
            }
            return X.array().rowwise() * ranges_.transpose().array() + mins_.transpose().array();
        }

    private:
        Vector mins_;
        Vector ranges_;
        bool fitted_ = false;
    };

}

#endif //AXML_SCALER_HPP