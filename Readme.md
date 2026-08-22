# AxML

AxML is a header-only C++20 machine learning library designed with a **Scikit-Learn style API** (`fit`, `predict`, `transform`).

## Features

- **Header-Only**: Simple integration—just include the headers in your C++ project.
- **Modern C++20**: Built using modern C++ features and standard practices.
- **Scikit-Learn Style Interface**: Intuitive parameter configuration and consistent model workflow.

### Available Algorithms & Modules

- **Classification**: Decision Tree, Random Forest, Logistic Regression, Support Vector Classifiers (Linear / RBF), Linear & Quadratic Discriminant Analysis (LDA / QDA), Naive Bayes.
- **Regression**: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree & Random Forest Regression.
- **Unsupervised & Transformers**: K-Means Clustering, PCA, Isolation Forest, Standard & Min-Max Scalers.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, MSE, MAE, R², Confusion Matrix.

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- CMake 3.20+ (for building tests/dev target)

## Quick Start

Since AxML is header-only, include `AxML` and its `deps` directory in your project's include paths.

```cpp
#include "AxML/AxML.hpp"
#include <iostream>

int main() {
    // 1. Configure hyperparameters
    AxML::DecisionTreeParams params;
    params.max_depth = 5;
    params.random_state = 42;

    // 2. Instantiate model
    AxML::DecisionTreeClassifier model(params);

    // 3. Fit model on feature matrix X and target y
    // model.fit(X, y);

    // 4. Predict on test data
    // AxML::VectorI preds = model.predict(X_test);

    return 0;
}
```

## Building Tests

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Project Structure

- `AxML/`: Core library headers and algorithm implementations.
- `deps/`: External header dependencies.
- `test/`: Unit tests, data generators, and boundary evaluation scripts.

## License

*TBD*
