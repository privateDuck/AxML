#pragma once

#ifndef AXML_MODEL_INDEX_HPP
#define AXML_MODEL_INDEX_HPP
#include <cstdint>

namespace AxML {
    enum ModelType : uint32_t {
        ID_LINEAR_REGRESSION = 0,
        ID_RIDGE_REGRESSION = 1,
        ID_LASSO_REGRESSION = 2,
        ID_ELASTICNET_REGRESSION = 3,
        ID_DT_REGRESSION = 4,
        ID_RF_REGRESSION = 5,
        ID_KNN_REGRESSION = 6,
        ID_MLP_REGRESSION = 7,

        ID_LOGISTIC_REGRESSION = 30,
        ID_DT_CLASSIFIER = 31,
        ID_RF_CLASSIFIER = 32,
        ID_LINEAR_SV_CLASSIFIER = 33,
        ID_RBF_SV_CLASSIFIER = 34,
        ID_KNN_CLASSIFIER = 35,
        ID_NB_CLASSIFIER = 36,
        ID_LDA_CLASSIFIER = 37,
        ID_QDA_CLASSIFIER = 38,
        ID_MLP_CLASSIFIER = 39,

        ID_ISOLATION_FOREST = 100,
        ID_DEEP_ISOLATION_FOREST = 101,
    };
}

#endif //AXML_MODEL_INDEX_HPP