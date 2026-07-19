#pragma once
#ifndef AXML_TENSOR3_HPP
#define AXML_TENSOR3_HPP

#include "common.hpp"

namespace AxML {
    struct Tensor3 {
        Tensor3();
        Tensor3(const i32 dim1, const i32 dim2, const i32 dim3): data(dim1 * dim2 * dim3), dim1_(dim1), dim2_(dim2) ,dim3_(dim3) {}
        Eigen::Map<MatrixR> operator()(const i32 i) {
            if (i < 0 || i >= dim1_) {
                throw std::out_of_range("Index out of range for Tensor3");
            }
            return Eigen::Map<MatrixR>(data.data() + static_cast<size_t>(i) * dim2_ * dim3_, dim2_, dim3_);
        }
    private:
        std::vector<Scalar> data;
        i32 dim1_{}, dim2_{}, dim3_{};
    };
}

#endif //AXML_TENSOR3_HPP