#pragma once
#ifndef AXML_RANDOMIZED_SVD_HPP
#define AXML_RANDOMIZED_SVD_HPP

#include <random>
#include "../common.hpp"

namespace AxML::utils {

    enum class RandomizedSVDMethod {
        Basic = 0,
        SubspaceIteration = 1
    };

    // Algorithm
    /*
     * A : Matrix to be decomposed
     * P <- Random Gaussian Matrix (n * k)
     * Z <- A * P
     * Q,R <- QR_Decomp(Z)
     * Y <- Q^T * A
     * Uy, Sz, Vt <- SVD(Y)
     * U <- Q * Uy
     * return U, Sy, Vt
     */
    class RandomizedSVD {
    public:
        explicit RandomizedSVD(const i32 k = 20, const RandomizedSVDMethod method = RandomizedSVDMethod::Basic)
            : k_(k), method_(method) {}

        std::tuple<MatrixC, Vector, MatrixC> compute(const MatrixC& A, const Scalar mean = 0.0, const Scalar std = 1.0, const i64 seed = 42) const {
            if (k_ <= 0 || k_ > A.cols()) {
                throw std::invalid_argument("k must be in the range (0, number of columns of A]");
            }

            std::mt19937 rng(seed);
            std::normal_distribution<Scalar> dist(mean, std);
            // Step 1: Generate a random Gaussian matrix P
            const MatrixC P = MatrixC::NullaryExpr(A.cols(), k_, [&]() {return dist(rng);});

            // Step 2: Compute Z = A * P
            MatrixC Z = A * P;

            // Step 3: Compute QR decomposition of Z
            const Eigen::HouseholderQR<MatrixC> qr(Z);
            MatrixC Q = qr.householderQ() * MatrixC::Identity(Z.rows(), k_);

            // Step 4: Compute Y = Q^T * A
            const MatrixC Y = Q.transpose() * A;

            // Step 5: Compute SVD of Y
            const Eigen::JacobiSVD<MatrixC> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const Vector S = svd.singularValues();
            const MatrixC U_y = svd.matrixU();
            MatrixC V_t = svd.matrixV().transpose();

            // Step 6: Compute U = Q * U_y
            MatrixC U = Q * U_y;

            return std::make_tuple(U, std::move(S), V_t);
        }
    private:
        i32 k_;
        RandomizedSVDMethod method_;
    };

}

#endif //AXML_RANDOMIZED_SVD_HPP