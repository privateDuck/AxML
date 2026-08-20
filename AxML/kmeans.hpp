#pragma once
#ifndef AXML_KMEANS_HPP
#define AXML_KMEANS_HPP

#include "base.hpp"
#include <random>

namespace AxML {

    enum class DistanceMetric : u32 {
        EUCLIDEAN = 0,
        MANHATTAN = 1,
        COSINE = 2,
    };

    class KMeans final : public Transformer {
    public:
        explicit KMeans(const i32 k, const DistanceMetric metric = DistanceMetric::EUCLIDEAN, const Scalar tol = 1e-6, const i64 random_state = 42)
            : tolerance_(tol), random_state_(random_state), metric_(metric), k_centroids_(k), is_fitted_(false) {}

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override { return is_fitted_; }
        uint64_t dims() const override { return 1; }

        void reset() override { is_fitted_ = false; final_centroids_.setZero(); }
        std::string name() const override { return "KMeans"; }
        uint32_t type_id() const override { return ID_KMEANS; }

        void fit(const MatrixR &X) override {
            const auto D = X.cols();
            const auto N = X.rows();
            MatrixR centroids(k_centroids_, D);
            std::mt19937_64 gen(random_state_);
            std::uniform_int_distribution<i32> uid(0, N - 1);
            const i32 first = uid(gen);
            centroids.row(0) = X.row(first);

            Vector closest_dist_sq = (X.rowwise() - centroids.row(0)).rowwise().squaredNorm();

            for (i32 i = 0; i < k_centroids_; ++i) {
                const Scalar sum = closest_dist_sq.sum();
                std::uniform_real_distribution<Scalar> dist(0, sum);
                const Scalar r = dist(gen);

                i32 idx = 0;
                Scalar cum = 0.0;
                for (i32 j = 0; j < N; ++j) {
                    cum += closest_dist_sq[i];
                    if (cum >= r) { idx = j; break; }
                }
                centroids.row(i) = X.row(idx);
                for (i32 j = 0; j < N; ++j) {
                    const Scalar d_sq = (X.row(j) - centroids.row(i)).squaredNorm();
                    if (d_sq < closest_dist_sq[j]) {
                        closest_dist_sq(i) = d_sq;
                    }
                }
            }

            // KMeans++ Initialization Complete
            // Elkan's Init
            std::vector upper_bounds(N, static_cast<Scalar>(0.0));
            std::vector lower_bounds(N * k_centroids_, static_cast<Scalar>(0.0));
            std::vector tight_bounds(N, false);

            for (int i = 0; i < N; ++i) {
                Scalar min_dist = std::numeric_limits<Scalar>::max();
                i32 best_c = 0;
                for (int c = 0; c < k_centroids_; ++c) {
                    const Scalar dist = (X.row(i) - centroids.row(c)).norm();
                    lower_bounds[i * k_centroids_ + c] = dist;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                labels[i] = best_c;
                upper_bounds[i] = min_dist;
                tight_bounds[i] = true;
            }

            std::vector<double> centroid_distances(k_centroids_ * k_centroids_, 0.0);
            std::vector<double> centroid_movement(k_centroids_, 0.0);

            // Main Loop
        }
        MatrixR transform(const MatrixR &X) const override;
        MatrixR fit_transform(const MatrixR &X) override;

    private:
        std::vector<i32> labels;
        MatrixR final_centroids_;
        Scalar tolerance_;
        u64 random_state_;
        DistanceMetric metric_;
        i32 k_centroids_;
        bool is_fitted_;
    };


}

#endif //AXML_KMEANS_HPP