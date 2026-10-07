#pragma once
#ifndef AXML_LABEL_ENCODER_HPP
#define AXML_LABEL_ENCODER_HPP

#include <vector>
#include <span>
#include <stdexcept>
#include "ankerl/unordered_dense.h"
#include "common.hpp"

namespace AxML {

class LabelEncoderInternal {
public:
    LabelEncoderInternal() = default;
    LabelEncoderInternal(const LabelEncoderInternal& other) = default;
    LabelEncoderInternal(LabelEncoderInternal&& other) noexcept : r_to_s_(std::move(other.r_to_s_)), s_to_r_(std::move(other.s_to_r_)) {}
    LabelEncoderInternal& operator=(const LabelEncoderInternal& other) = default;
    LabelEncoderInternal& operator=(LabelEncoderInternal&& other) noexcept {
        r_to_s_ = std::move(other.r_to_s_);
        s_to_r_ = std::move(other.s_to_r_);
        return *this;
    }

    void fit(const std::span<const i32> input) {
        r_to_s_.clear();
        s_to_r_.clear();

        // Pre-allocate to minimize hash collisions and reallocations during population
        r_to_s_.reserve(input.size());
        s_to_r_.reserve(input.size());

        i32 current_s_label = 0;

        for (const i32 val : input) {
            // If the R value is not yet in our map, assign it the next S label
            if (r_to_s_.find(val) == r_to_s_.end()) {
                r_to_s_[val] = current_s_label++;
                s_to_r_.push_back(val);
            }
        }
    }

    [[nodiscard]] std::vector<i32> transform(const std::span<const i32> input) const {
        std::vector<i32> encoded;
        encoded.reserve(input.size());

        for (const i32 val : input) {
            auto it = r_to_s_.find(val);
            if (it == r_to_s_.end()) {
                throw std::invalid_argument("Unseen R space value encountered during transform.");
            }
            encoded.push_back(it->second);
        }

        return encoded;
    }

    [[nodiscard]] std::vector<Scalar> transform_to_float(const std::span<const i32> input) const {
        std::vector<Scalar> encoded;
        encoded.reserve(input.size());

        for (const i32 val : input) {
            auto it = r_to_s_.find(val);
            if (it == r_to_s_.end()) {
                throw std::invalid_argument("Unseen R space value encountered during transform.");
            }
            // Cast the integer label to a Scalar representation
            encoded.push_back(static_cast<Scalar>(it->second));
        }

        return encoded;
    }

    [[nodiscard]] std::vector<Scalar> transform_to_onehot_col_major(const std::span<const i32> input) const {
        const auto rows = input.size();
        const auto cols = num_unique_labels();
        std::vector encoded(cols * rows, 0.0);

        for (i32 row = 0; row < rows; ++row) {
            auto it = r_to_s_.find(input[row]);
            if (it == r_to_s_.end()) {
                throw std::invalid_argument("Unseen R space value encountered during transform.");
            }
            const auto index = it->second * rows + row;
            // Cast the integer label to a Scalar representation
            encoded[index] = 1.0;
        }

        return encoded;
    }

    [[nodiscard]] std::vector<Scalar> transform_to_onehot_row_major(const std::span<const i32> input) const {
        const auto rows = input.size();
        const auto cols = num_unique_labels();
        std::vector encoded(cols * rows, 0.0);

        for (i32 row = 0; row < rows; ++row) {
            auto it = r_to_s_.find(input[row]);
            if (it == r_to_s_.end()) {
                throw std::invalid_argument("Unseen R space value encountered during transform.");
            }
            const auto index = row * cols + it->second;
            // Cast the integer label to a Scalar representation
            encoded[index] = 1.0;
        }

        return encoded;
    }

    [[nodiscard]] i32 inverse_transform(const i32 s_label) const {
        // Bounds checking
        if (s_label < 0 || static_cast<size_t>(s_label) >= s_to_r_.size()) {
            throw std::out_of_range("S space label out of bounds.");
        }

        return s_to_r_[s_label];
    }

    [[nodiscard]] std::vector<i32> inverse_transform(const std::span<const i32> input) const {
        std::vector<i32> decoded;
        decoded.reserve(input.size());

        for (const i32 s_label : input) {
            decoded.push_back(inverse_transform(s_label));
        }

        return decoded;
    }

    [[nodiscard]] i32 num_unique_labels() const {
        return static_cast<i32>(r_to_s_.size());
    }

private:
    ankerl::unordered_dense::map<i32, i32> r_to_s_;
    std::vector<i32> s_to_r_;
};

}

#endif //AXML_LABEL_ENCODER_HPP