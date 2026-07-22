#ifndef AXML_COMMON_HPP
#define AXML_COMMON_HPP

#include <Eigen/Dense>
#include "model_index.hpp"

namespace AxML {
    using Scalar = double;
    using i32 = int32_t;
    using i64 = int64_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using MatrixR = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // Eigen default
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>; // Eigen default
    using VectorI = Eigen::Matrix<i32, Eigen::Dynamic, 1, Eigen::ColMajor>; // Eigen default
    using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::RowMajor>;

    inline bool scmp(const Scalar a, const Scalar b) {
        constexpr Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        float diff = std::abs(a - b);
        if (diff <= epsilon) return true;
        return diff <= (epsilon * std::max(std::abs(a), std::abs(b)));
    }
}
#endif //AXML_COMMON_HPP