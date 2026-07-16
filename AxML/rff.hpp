#pragma once

#ifndef AXML_RFF_HPP
#define AXML_RFF_HPP

#include <random>
#include "common.hpp"

namespace AxML {

    class RandomFourierFeatures {
    public:
        RandomFourierFeatures() = default;

        RandomFourierFeatures(const uint32_t input_dims, const uint32_t dimensions, const Scalar gamma)
            : gamma_(gamma), input_dims_(input_dims), D_(dimensions) {

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, std::sqrt(2 * gamma_));
            std::uniform_real_distribution<Scalar> dist_unif(0.0, 2.0 * pi);

            W_ = MatrixR::NullaryExpr(D_, input_dims_, [&](){return dist(gen);});
            b_ = Vector::NullaryExpr(D_, [&](){return dist_unif(gen);});
        }

        void generate(const uint32_t input_dims, const uint32_t dimensions, const Scalar gamma, const uint64_t seed = 42) {
            D_ = dimensions;
            input_dims_ = input_dims;
            gamma_ = gamma;
            std::mt19937 gen(seed);
            std::normal_distribution<Scalar> dist_norm(0.0, std::sqrt(2 * gamma_));
            std::uniform_real_distribution<Scalar> dist_unif(0.0, 2.0 * pi);

            W_ = MatrixR::NullaryExpr(D_, input_dims_, [&](){return dist_norm(gen);});
            b_ = Vector::NullaryExpr(D_, [&](){return dist_unif(gen);});
        }

        MatrixR transform(const MatrixR& X) const {
            if (X.cols() != input_dims_) {
                throw std::invalid_argument("Input dimensions do not match the expected input dimensions.");
            }

            MatrixR Z = (X * W_.transpose()).rowwise() + b_.transpose();
            Z = Z.array().cos();
            Z *= std::sqrt(2.0 / D_);
            return Z;
        }

        void reset() {
            W_.setZero();
            b_.setZero();
            gamma_ = 1.0;
            input_dims_ = 0;
            D_ = 0;
        }

    private:
        MatrixR W_;
        Vector b_;
        Scalar gamma_;
        uint32_t input_dims_;
        uint32_t D_;

        static constexpr Scalar pi = 3.1415926535897932384626433832795;
    };

}

#endif //AXML_RFF_HPP