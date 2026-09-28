#pragma once
#ifndef AXML_PARAM_CONFIG_HPP
#define AXML_PARAM_CONFIG_HPP

#include "common.hpp"
#include <string>


namespace AxML {

/**
 * @brief Configuration parameters for the Logistic Regression model.
 */
struct LogisticRegressionParams {
  /** @brief Regularization type: "none", "l1", "l2", or "elastic_net" */
  std::string regularization = "l1";
  /** @brief Maximum number of iterations for the solver */
  int max_iterations = 10000;
  /** @brief L1 regularization strength (used if regularization is "l1" or "elastic_net") */
  Scalar l1_regularization = 0.01;
  /** @brief L2 regularization strength (used if regularization is "l2" or "elastic_net") */
  Scalar l2_regularization = 0.01;
  /** @brief Tolerance for stopping criteria */
  Scalar tolerance = 1e-6;

  int get_reg() const {
    if (regularization == "none")
      return 0;
    if (regularization == "l1")
      return 1;
    if (regularization == "l2")
      return 2;
    if (regularization == "elastic_net")
      return 3;
    throw std::invalid_argument("Invalid regularization type. (Only 'none', "
                                "'l1', 'l2', and 'elastic_net' are allowed");
  }
};

/**
 * @brief Configuration parameters for the Decision Tree model.
 */
struct DecisionTreeParams {
  /** @brief Maximum depth of the tree. -1 means unlimited depth. */
  i32 max_depth = -1;
  /** @brief Minimum number of samples required to split an internal node */
  i32 min_samples_split = 2;
  /** @brief Minimum number of samples required to be at a leaf node */
  i32 min_samples_leaf = 1;
  /** @brief The number of features to consider when looking for the best split: "all", "sqrt", "log2", or an integer string */
  std::string max_features = "sqrt"; // all, sqrt, log2, integer
  /** @brief Seed used by the random number generator */
  u64 random_state = 42;

  [[nodiscard]] static std::string get_mf_str(const i32 mf) {
    switch (mf) {
    case -1:
      return "all";
    case -2:
      return "sqrt";
    case -3:
      return "log2";
    default:
      return std::to_string(mf);
    }
  }

private:
  friend class DecisionTreeClassifier;
  friend class DecisionTreeRegressor;
  [[nodiscard]] i32 get_mf() const {
    if (max_features == "all")
      return -1;
    if (max_features == "sqrt")
      return -2;
    if (max_features == "log2")
      return -3;
    try {
      return std::stoi(max_features);
    } catch (const std::exception &) {
      throw std::invalid_argument("Invalid max_features value. (Only 'all', "
                                  "'sqrt', 'log2', or an integer are allowed)");
    }
  }
};

/**
 * @brief Configuration parameters for the Random Forest model.
 */
struct RandomForestParams {
  /** @brief The number of trees in the forest */
  i32 n_estimators = 100;
  /** @brief Maximum depth of the tree. -1 means unlimited depth. */
  i32 max_depth = -1;
  /** @brief Minimum number of samples required to split an internal node */
  i32 min_samples_split = 2;
  /** @brief Minimum number of samples required to be at a leaf node */
  i32 min_samples_leaf = 1;
  /** @brief The number of features to consider when looking for the best split: "all", "sqrt", "log2", or an integer string */
  std::string max_features = "sqrt";
  /** @brief Whether bootstrap samples are used when building trees */
  bool enable_bootstrap = true;
  /** @brief The size of the bootstrap sample. -1 means sample size is equal to the number of input samples */
  i32 bootstrap_size = -1;
  /** @brief The number of threads to run in parallel. 0 means main thread. -1 means use all available processors */
  i32 n_threads = -1; // -1: all, 1: strict single threaded, n: n threads
  /** @brief Seed used by the random number generator */
  u64 random_state = 42;

private:
  friend class RandomForestClassifier;
  friend class RandomForestRegressor;

  [[nodiscard]] static std::string get_mf_str(const i32 mf) {
    switch (mf) {
    case -1:
      return "all";
    case -2:
      return "sqrt";
    case -3:
      return "log2";
    default:
      return std::to_string(mf);
    }
  }

  [[nodiscard]] i32 get_mf() const {
    if (max_features == "all")
      return -1;
    if (max_features == "sqrt")
      return -2;
    if (max_features == "log2")
      return -3;
    try {
      return std::stoi(max_features);
    } catch (const std::exception &) {
      throw std::invalid_argument("Invalid max_features value. (Only 'all', "
                                  "'sqrt', 'log2', or an integer are allowed)");
    }
  }
};

/**
 * @brief Configuration parameters for the Linear Support Vector Classification (LinearSVC) model.
 */
struct LinearSVCParams {
  /** @brief Regularization parameter. The strength of the regularization is inversely proportional to C */
  Scalar C = 1.0;
  /** @brief Maximum number of iterations for the solver */
  i32 max_iter = 10000;
  /** @brief Tolerance for stopping criteria */
  Scalar tolerance = 1e-6;
};

/**
 * @brief Configuration parameters for the Support Vector Classification (SVC) model with Radial Basis Function (RBF) kernel.
 */
struct RBFSVCParams {
  /** @brief Kernel coefficient for 'rbf' */
  Scalar gamma = 1.0;
  /** @brief Regularization parameter. The strength of the regularization is inversely proportional to C */
  Scalar C = 1.0;
  /** @brief Number of Random Fourier features to approximate the RBF kernel */
  u32 rf_features = 100;
  /** @brief Maximum number of iterations for the solver */
  i32 max_iter = 10000;
  /** @brief Tolerance for stopping criteria */
  Scalar tolerance = 1e-6;
};


  struct LinearRegressorParams {
    std::string regularization = "none";
    i32 max_iter = 10000;
    Scalar tolerance = 1e-6;
    Scalar l1_reg = 0.01;
    Scalar l2_reg = 0.01;
  };

struct MLPParams {
  std::vector<i32> hidden_layers = {100};
  std::string activation = "relu"; // relu, tanh, sigmoid, softmax (non-trainable), linear
  std::string solver = "sgd"; // adam, sgd
  std::string regularization = "none";
  u64 random_state = 42;
  Scalar learning_rate = 1e-3; // init value
  Scalar beta_1 = 0.9; // adam
  Scalar beta_2 = 0.99; // adam
  Scalar epsilon = 1e-6; // adam
  Scalar momentum = 0.0; // sgd
  Scalar tolerance = 1e-6;
  i32 batch_size = 64;
  i32 max_iter = 10000;
  std::function<void(Scalar, Scalar, i32)> callback_fn; // Called per batch. (data_loss, regularization_loss, iteration)
};

} // namespace AxML

#endif // AXML_PARAM_CONFIG_HPP