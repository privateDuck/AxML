#pragma once
#ifndef AXML_METRICS_HPP
#define AXML_METRICS_HPP

#include "common.hpp"
#include "label_encoder.hpp"


namespace AxML::metrics {
    class Classifier;

    // Need a better name
    struct PredictionCounts {
        Scalar weight{};
        i32 TP; // true positives
        i32 TN; // true negatives
        i32 FP; // false positives
        i32 FN; // false negatives
        PredictionCounts() : weight(0.0), TP(0), TN(0), FP(0), FN(0) {}
    };

    // Needs a better name
    inline std::vector<PredictionCounts> multi_count(const LabelEncoderInternal& enc, const Eigen::Ref<const VectorI>& preds, const Eigen::Ref<const VectorI>& truths) {
        std::vector<PredictionCounts> counts(enc.num_unique_labels());
        const auto enc_preds = enc.transform(std::span(preds.data(), preds.size()));
        const auto enc_truths = enc.transform(std::span(truths.data(), truths.size()));

        for (i32 class_ = 0; class_ < enc.num_unique_labels(); ++class_) {
            // Whole reason to encode again is to guarantee that classes are 0 indexed
            // We would need to collect unique values anyway to ensure that we have all classes, so this is a better approach
            auto&[weight, TP, TN, FP, FN] = counts[class_];

            for (i32 i = 0; i < preds.size(); ++i) {
                const i32 true_class_ = enc_truths[i];
                const i32 pred_class_ = enc_preds[i];

                if (pred_class_ == class_ && true_class_ == class_) {
                    ++TP;
                    weight += 1.0;
                } else if (pred_class_ == class_ && true_class_ != class_) {
                    ++FP;
                } else if (pred_class_ != class_ && true_class_ != class_) {
                    ++TN;
                } else if (pred_class_ != class_ && true_class_ == class_) {
                    ++FN;
                    weight += 1.0;
                }
                weight /= static_cast<Scalar>(preds.size());
            }
        }
        return counts;
    }

    enum class AverageMethod : u32 {
        Macro = 0,
        Micro = 1,
        Weighted = 2,
    };

    inline Scalar accuracy(const Eigen::Ref<const VectorI>& preds, const Eigen::Ref<const VectorI>& truths) {
        if (preds.size() != truths.size()) {
            throw std::invalid_argument("Predictions and truths must have the same size.");
        }
        i32 correct = 0;
        for (i32 i = 0; i < preds.size(); ++i) {
            if (preds(i) == truths(i)) {
                ++correct;
            }
        }
        return static_cast<Scalar>(correct) / static_cast<Scalar>(preds.size());
    }

    inline Scalar accuracy(const PredictionCounts& counts) {
        if (counts.TP + counts.TN + counts.FP + counts.FN == 0) return 0.0;
        return static_cast<Scalar>(counts.TP + counts.TN) / static_cast<Scalar>(counts.TP + counts.TN + counts.FP + counts.FN);
    }

    inline Scalar precision(const PredictionCounts& counts) {
        const i32 TP = counts.TP;
        const i32 FP = counts.FP;
        if (TP + FP == 0) return 0.0;
        return static_cast<Scalar>(TP) / static_cast<Scalar>(TP + FP);
    }

    inline Scalar recall(const PredictionCounts& counts) {
        const i32 TP = counts.TP;
        const i32 FN = counts.FN;
        if (TP + FN == 0) return 0.0;
        return static_cast<Scalar>(TP) / static_cast<Scalar>(TP + FN);
    }

    inline Scalar f1(const PredictionCounts& counts) {
        if (counts.TP + counts.FP == 0 || counts.TP + counts.FN == 0) return 0.0;
        const auto prec = precision(counts);
        const auto rec = recall(counts);
        return 2.0 * (prec * rec) / (prec + rec); // Add small epsilon to avoid division by zero
    }

    inline Scalar average_accuracy(const std::vector<PredictionCounts>& counts) {
        Scalar sum_tp = 0.0;
        Scalar sum_fn = 0.0;
        for (const auto& count : counts) {
            sum_tp += count.TP;
            sum_fn += count.FN;
        }
        return sum_tp / (sum_tp + sum_fn);
    }

    inline Scalar average_precision(const std::vector<PredictionCounts>& counts, const AverageMethod method = AverageMethod::Macro) {
        if (counts.size() == 0) return 0.0;
        const Scalar n_classes = static_cast<Scalar>(counts.size());
        if (const i32 total_samples = counts[0].TP + counts[0].TN + counts[0].FP + counts[0].FN; total_samples == 0) return 0.0;

        if (method == AverageMethod::Macro) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += precision(count);
            }
            return sum / n_classes;
        }
        if (method == AverageMethod::Micro) {
            Scalar sum_tp = 0.0, sum_fp = 0.0;
            for (const auto& count : counts) {
                sum_tp += count.TP;
                sum_fp += count.FP;
            }
            return sum_tp / (sum_tp + sum_fp);
        }
        if (method == AverageMethod::Weighted) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += count.weight * precision(count);
            }
            return sum / n_classes;
        }
        return 0.0; // Default return value if method is not recognized
    }

    inline Scalar average_recall(const std::vector<PredictionCounts>& counts, const AverageMethod method = AverageMethod::Macro) {
        if (counts.size() == 0) return 0.0;
        const Scalar n_classes = static_cast<Scalar>(counts.size());
        if (const i32 total_samples = counts[0].TP + counts[0].TN + counts[0].FP + counts[0].FN; total_samples == 0) return 0.0;

        if (method == AverageMethod::Macro) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += recall(count);
            }
            return sum / n_classes;
        }
        if (method == AverageMethod::Micro) {
            Scalar sum_tp = 0.0, sum_fn = 0.0;
            for (const auto& count : counts) {
                sum_tp += count.TP;
                sum_fn += count.FN;
            }
            return sum_tp / (sum_tp + sum_fn);
        }
        if (method == AverageMethod::Weighted) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += count.weight * recall(count);
            }
            return sum / n_classes;
        }
        return 0.0; // Default return value if method is not recognized
    }

    inline Scalar average_f1(const std::vector<PredictionCounts>& counts, const AverageMethod method = AverageMethod::Macro) {
        if (counts.size() == 0) return 0.0;
        const Scalar n_classes = static_cast<Scalar>(counts.size());
        if (const i32 total_samples = counts[0].TP + counts[0].TN + counts[0].FP + counts[0].FN; total_samples == 0) return 0.0;

        if (method == AverageMethod::Macro) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += f1(count);
            }
            return sum / n_classes;
        }
        if (method == AverageMethod::Micro) {
            Scalar sum_tp = 0.0, sum_fn = 0.0, sum_fp = 0.0;
            for (const auto& count : counts) {
                sum_tp += count.TP;
                sum_fn += count.FN;
                sum_fp += count.FP;
            }
            return 2.0 * sum_tp / (2.0 * sum_tp + sum_fp + sum_fn);
        }
        if (method == AverageMethod::Weighted) {
            Scalar sum = 0.0;
            for (const auto& count : counts) {
                sum += count.weight * f1(count);
            }
            return sum / n_classes;
        }
        return 0.0; // Default return value if method is not recognized
    }

    // Smallest class goes on the first row/column
    // Filled in sorted order of the size of the class
    inline MatrixC confusion_matrix(const LabelEncoderInternal& encoder, const VectorI& preds, const VectorI& actual) {
        const auto enc_preds = encoder.transform(std::span(preds.data(), preds.size()));
        const auto enc_actual = encoder.transform(std::span(actual.data(), actual.size()));
        const i32 n_classes = encoder.num_unique_labels();
        MatrixC cm = MatrixC::Zero(n_classes, n_classes);
        for (i32 i = 0; i < preds.size(); ++i) {
            const i32 row = enc_actual[i];
            const i32 col = enc_preds[i];
            cm(row, col) += 1.0;
        }
        return cm;
    }

    class ClassificationReport {
    public:
        ClassificationReport() = default;

        Scalar accuracy_score() const { return average_accuracy(counts_); }
        Scalar precision_score(const AverageMethod method = AverageMethod::Macro) const { return average_precision(counts_, method); }
        Scalar recall_score(const AverageMethod method = AverageMethod::Macro) const { return average_recall(counts_, method); }
        Scalar f1_score(const AverageMethod method = AverageMethod::Macro) const { return average_f1(counts_, method); }
        Scalar balanced_accuracy() const { return average_recall(counts_, AverageMethod::Macro); }
        const MatrixC& get_confusion_matrix() const { return cm_; }

    private:
        friend Classifier;

        void score(const LabelEncoderInternal& encoder, const Eigen::Ref<const VectorI>& preds, const Eigen::Ref<const VectorI>& actual) {
            counts_ = multi_count(encoder, preds, actual);
            cm_ = confusion_matrix(encoder, preds, actual);
        }

        std::vector<PredictionCounts> counts_;
        MatrixC cm_;
    };

    //-------------------------------------//
    // Regression Metrics //

    inline Scalar R2(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual) {
        if (preds.size() != actual.size()) {
            throw std::invalid_argument("Predictions must have the same size.");
        }
        const Scalar mean_actual = actual.mean();
        Scalar ss_total = 0.0;
        Scalar ss_residual = 0.0;
        for (i32 i = 0; i < preds.size(); ++i) {
            ss_total += (actual(i) - mean_actual) * (actual(i) - mean_actual);
            ss_residual += (actual(i) - preds(i)) * (actual(i) - preds(i));
        }
        return 1.0 - (ss_residual / ss_total);
    }

    inline Scalar AdjR2(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual, const i32 predictors) {
        if (preds.size() != actual.size()) {
            throw std::invalid_argument("Predictions must have the same size.");
        }
        const Scalar r2 = R2(preds, actual);
        const Scalar num = static_cast<Scalar>(actual.size() - 1);
        const Scalar denom = static_cast<Scalar>(actual.size() - predictors - 1);
        return 1.0 - (1.0 - r2) * ( num/denom );
    }

    inline Scalar MeanAbsoluteError(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual) {
        if (preds.size() != actual.size()) {
            throw std::invalid_argument("Predictions must have the same size.");
        }
        Scalar abs_sum = 0.0;
        for (i32 i = 0; i < preds.size(); ++i) {
            abs_sum += std::abs(preds(i) - actual(i));
        }
        return abs_sum / static_cast<Scalar>(preds.size());
    }

    inline Scalar MeanSquaredError(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual) {
        if (preds.size() != actual.size()) {
            throw std::invalid_argument("Predictions must have the same size.");
        }
        Scalar sq_sum = 0.0;
        for (i32 i = 0; i < preds.size(); ++i) {
            const Scalar diff = (preds(i) - actual(i));
            sq_sum += diff * diff;
        }
        return sq_sum / static_cast<Scalar>(preds.size());
    }

    inline Scalar RootMeanSquaredError(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual) {
        return std::sqrt(MeanSquaredError(preds, actual));
    }

    inline Scalar MeanAbsolutePercentageError(const Eigen::Ref<const Vector>& preds, const Eigen::Ref<const Vector>& actual) {
        if (preds.size() != actual.size()) {
            throw std::invalid_argument("Predictions must have the same size.");
        }
        Scalar abs_sum = 0.0;
        for (i32 i = 0; i < preds.size(); ++i) {
            abs_sum += std::abs((preds(i) - actual(i)) / actual(i));
        }
        return abs_sum / static_cast<Scalar>(preds.size());
    }

}

#endif //AXML_METRICS_HPP
