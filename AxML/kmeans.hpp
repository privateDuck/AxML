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
        explicit KMeans(const i32 k, const DistanceMetric metric = DistanceMetric::EUCLIDEAN, const Scalar tol = 1e-6, const i32 max_iter = 10000, const i64 random_state = 42)
            : tolerance_(tol), random_state_(random_state), metric_(metric), max_iter_(max_iter), k_centroids_(k), is_fitted_(false) {}

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        bool is_fitted() const override { return is_fitted_; }
        uint64_t dims() const override { return 1; }

        void reset() override { is_fitted_ = false; final_centroids_.setZero(); }
        std::string name() const override { return "KMeans"; }
        uint32_t type_id() const override { return ID_KMEANS; }

        void fit(const ConstMatRRef &X) override {
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
                labels_[i] = best_c;
                upper_bounds[i] = min_dist;
                tight_bounds[i] = true;
            }

            std::vector<double> centroid_distances(k_centroids_ * k_centroids_, 0.0);
            std::vector<double> centroid_movement(k_centroids_, 0.0);

            MatrixR new_centroids(k_centroids_, D);
            std::vector<i32> counts(k_centroids_, 0);

            // Main Loop
            for (i32 iter = 0; iter < max_iter_; ++iter) {
                // Compute centroid distances
                for (i32 c1 = 0; c1 < k_centroids_; ++c1) {
                    for (i32 c2 = c1 + 1; c2 < k_centroids_; ++c2) {
                        const Scalar dist = (centroids.row(c1) - centroids.row(c2)).norm();
                        centroid_distances[c1 * k_centroids_ + c2] = dist;
                        centroid_distances[c2 * k_centroids_ + c1] = dist;
                    }
                }

                for (i32 i = 0; i < N; ++i) {
                    const i32 c = labels_[i];
                    Scalar min_dist = upper_bounds[i];
                    if (!tight_bounds[i]) {
                        min_dist = (X.row(i) - centroids.row(c)).norm();
                        upper_bounds[i] = min_dist;
                        lower_bounds[i * k_centroids_ + c] = min_dist;
                        tight_bounds[i] = true;
                    }

                    i32 new_c = c;
                    for (i32 j = 0; j < k_centroids_; ++j) {
                        if (j == c) continue;

                        // Pruning 1: Half-distance to other centroid
                        if (min_dist <= centroid_distances[c * k_centroids_ + j]) continue;
                        // Pruning 2: Lower bound
                        if (min_dist <= lower_bounds[i * k_centroids_ + j]) continue;

                        const Scalar dist_ij = (X.row(i) - centroids.row(j)).norm();
                        lower_bounds[i * k_centroids_ + j] = dist_ij;

                        if (dist_ij < min_dist) {
                            min_dist = dist_ij;
                            new_c = j;
                        }
                    }

                    if (new_c != c) {
                        labels_[i] = new_c;
                        upper_bounds[i] = min_dist;
                    }
                }

                new_centroids.setZero();
                std::ranges::fill(counts, 0.0);

                for (int i = 0; i < N; ++i) {
                    int c = labels_[i];
                    counts[c]++;
                    new_centroids.row(c) += X.row(i);
                }

                Scalar max_movements = 0.0;
                for (int c = 0; c < k_centroids_; ++c) {
                    if (counts[c] > 0) {
                        new_centroids.row(c) /= static_cast<Scalar>(counts[c]);
                    } else {
                        new_centroids.row(c) = centroids.row(c);
                    }
                    const Scalar movement = (centroids.row(c) - new_centroids.row(c)).norm();
                    centroid_movement[c] = movement;
                    max_movements = std::max(max_movements, movement);
                }

                for (i32 i = 0; i < N; ++i) {
                    const i32 c = labels_[i];
                    upper_bounds[i] += centroid_movement[c];
                    tight_bounds[i] = false;

                    for (int j = 0; j < k_centroids_; ++j) {
                        lower_bounds[i * k_centroids_ + j] = std::max(lower_bounds[i * k_centroids_ + j] - centroid_movement[j], static_cast<Scalar>(0.0));
                    }
                }

                centroids = std::move(new_centroids);

                if (max_movements < tolerance_) break;
            }

            final_centroids_ = std::move(centroids);
            is_fitted_ = true;
            labels_.clear();
            labels_.shrink_to_fit();
        }
        MatrixR transform(const ConstMatRRef &X) const override {
            if (!is_fitted_) {
                throw std::logic_error("Transform is not fitted.");
            }
            MatrixR preds(X.rows(), 1);
            for (i32 i = 0; i < X.rows(); ++i) {
                Scalar min_sq_dist = std::numeric_limits<Scalar>::max();
                i32 best_c = 0;

                for (int c = 0; c < k_centroids_; ++c) {
                    const Scalar dist = (X.row(i) - final_centroids_.row(c)).squaredNorm();
                    if (dist < min_sq_dist) {
                        min_sq_dist = dist;
                        best_c = c;
                    }
                }
                preds(i, 0) = static_cast<Scalar>(best_c);
            }
            return preds;
        }
        MatrixR fit_transform(const ConstMatRRef &X) override {
            fit(X);
            MatrixR preds(X.rows(), 1);
            for (i32 i = 0; i < X.rows(); ++i) {
                preds(i, 0) = static_cast<Scalar>(labels_[i]);
            }
            labels_.clear();
            labels_.shrink_to_fit();
            return preds;
        }

    private:
        std::vector<i32> labels_;
        MatrixR final_centroids_;
        Scalar tolerance_;
        u64 random_state_;
        DistanceMetric metric_;
        i32 max_iter_;
        i32 k_centroids_;
        bool is_fitted_;
    };


}

#endif //AXML_KMEANS_HPP