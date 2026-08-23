#ifndef AXML_COMMON_HPP
#define AXML_COMMON_HPP

#include <Eigen/Dense>
#include "model_index.hpp"
#define AXML_MATR_ASSERT(Derived) static_assert(std::is_same_v<typename Derived::Scalar, Scalar>); static_assert(Derived::IsRowMajor,"row-major storage preferred");
#define AXML_MATC_ASSERT(Derived) static_assert(std::is_same_v<typename Derived::Scalar, Scalar>); static_assert(Derived::IsColumnMajor,"column-major storage preferred");
#define AXML_IVEC_ASSERT(Derived) static_assert(std::is_same_v<typename Derived::Scalar, i32>); static_assert(Derived::ColsAtCompileTime == 1);
#define AXML_FVEC_ASSERT(Derived) static_assert(std::is_same_v<typename Derived::Scalar, Scalar>); static_assert(Derived::ColsAtCompileTime == 1);

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

    using ConstMatRRef = Eigen::Ref<const MatrixR>;
    using ConstMatCRef = Eigen::Ref<const MatrixC>;
    using ConstVecRef = Eigen::Ref<const Vector>;
    using ConstVecIRef = Eigen::Ref<const VectorI>;

    using MapMatrixR      = Eigen::Map<MatrixR>;
    using MapConstMatrixR = Eigen::Map<const MatrixR>;
    using MapMatrixC      = Eigen::Map<MatrixC>;
    using MapConstMatrixC = Eigen::Map<const MatrixC>;
    using MapVector        = Eigen::Map<Vector>;
    using MapConstVector   = Eigen::Map<const Vector>;
    using MapVectorI       = Eigen::Map<VectorI>;
    using MapConstVectorI  = Eigen::Map<const VectorI>;

    inline bool scmp(const Scalar a, const Scalar b) {
        constexpr Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        const Scalar diff = std::abs(a - b);
        if (diff <= epsilon) return true;
        return diff <= (epsilon * std::max(std::abs(a), std::abs(b)));
    }
}
#endif //AXML_COMMON_HPP