#pragma once
#ifndef AXML_ESTIMATOR_HPP
#define AXML_ESTIMATOR_HPP

#include <span>
#include "common.hpp"

namespace AxML {
    class OutputArchive;
    class InputArchive;

    class Estimator {
    public:
        virtual ~Estimator() = default;

        virtual void fit(const MatrixR& X, const Vector& y) {
            if(X.rows() != y.size()) {
                throw std::logic_error("Number of rows in X must match size of y");
            }
            fit_impl(X, y);
        }

        virtual void save(OutputArchive& ar) const = 0;
        virtual void load(InputArchive& ar) = 0;

        virtual std::unique_ptr<Estimator> clone() const = 0;

        virtual bool is_fitted() const = 0;
        virtual uint64_t dims() const = 0;
        virtual void reset() = 0;

        virtual std::string name() const = 0;
        virtual uint32_t type_id() const = 0;

    protected:
        virtual void fit_impl(const MatrixR& X, const Vector& y) = 0;
    };


    class Classifier : public Estimator {
    public:
        virtual Vector predict(const MatrixR& X) const = 0;
        virtual bool supports_predict_proba() const noexcept { return false; }
        virtual MatrixR predict_proba(const MatrixR& X) const {
            throw std::logic_error("predict_proba not supported by this classifier");
        }
    };


    class Regressor : public Estimator {
    public:
        virtual Vector predict(const MatrixR& X) const = 0;
    };


    class Transformer : public Estimator {
    public:
        virtual MatrixR transform(const MatrixR& X) const = 0;
    };


    class OutputArchive {
    public:
        virtual ~OutputArchive() = default;
        virtual void write_u32(std::uint32_t) = 0;
        virtual void write_f64(double) = 0;
        virtual void write_string(std::string_view) = 0;
        virtual void write_bytes(std::span<const std::byte>) = 0;
    };

    class InputArchive {
    public:
        virtual ~InputArchive() = default;
        virtual uint32_t read_u32() = 0;
        virtual double read_f64() = 0;
        virtual std::string read_string() = 0;
        virtual std::span<std::byte> read_bytes() = 0;
    };

}
#endif //AXML_ESTIMATOR_HPP