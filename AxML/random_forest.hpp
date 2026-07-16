#pragma once

#ifndef AXML_RANDOM_FOREST_HPP
#define AXML_RANDOM_FOREST_HPP

#include "decision_tree.hpp"

namespace AxML {

    class RandomForestClassifier final : public Classifier {
    public:
        explicit RandomForestClassifier(
            const int32_t n_estimators = 100, const int32_t max_depth = -1,
            const int32_t min_samples_split = 2, const int32_t min_samples_leaf = 1,
            const MaxFeatures max_features = MaxFeatures::Sqrt, const bool bootstrap = true,
            const uint64_t random_state = 42)
            : random_state_(random_state), n_trees_(n_estimators), max_depth_(max_depth),
              min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf),
              max_features_(static_cast<int32_t>(max_features)), bootstrap_(bootstrap) {}

        void fit(const MatrixR &X, const Vector &y) override;
        Vector predict(const MatrixR &X) const override;
        MatrixR predict_proba(const MatrixR &X) const override;

        void save(OutputArchive &ar) const override;
        void load(InputArchive &ar) override;
        std::unique_ptr<Estimator> clone() const override;

        void reset() override;

        std::string name() const override;
        uint32_t type_id() const override;

    protected:
        void fit_impl(const MatrixR &X, const Vector &y) override;

    private:
        std::vector<DecisionTreeClassifier> trees_;
        uint64_t random_state_;
        int32_t n_trees_;
        int32_t max_depth_;
        int32_t min_samples_split_;
        int32_t min_samples_leaf_;
        int32_t max_features_;
        bool bootstrap_;
    };

}

#endif //AXML_RANDOM_FOREST_HPP