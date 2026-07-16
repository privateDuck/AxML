#ifndef AXML_COMMON_HPP
#define AXML_COMMON_HPP

#include <Eigen/Dense>
#include "model_index.hpp"

namespace AxML {
    using Scalar = double;
    using MatrixR = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorR = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::AutoAlign>;
}
#endif //AXML_COMMON_HPP