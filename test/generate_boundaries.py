import numpy as np
import warnings
from sklearn.datasets import make_blobs

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

warnings.filterwarnings("ignore")

def cpp_float_vec(arr):
    return "{" + ", ".join(f"{x:.4f}f" for x in np.asarray(arr).flatten()) + "}"

def cpp_int_vec(arr):
    return "{" + ", ".join(str(int(x)) for x in np.asarray(arr).flatten()) + "}"

def cpp_string(val):
    return f'"{val}"' if isinstance(val, str) else str(val)

def main():
    # 1. Generate 2D Multiclass Dataset
    X_train, y_train = make_blobs(n_samples=150, centers=3, n_features=2,
                                  cluster_std=2.2, random_state=42)

    # 2. Generate 2D Grid
    h = 0.15
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    print(f"Grid shape: {xx.shape}, Total grid points: {X_grid.shape[0]}")

    # 3. Define Models and Hyperparameters
    # Format: "ModelName": (ModelClass, [List of configs])
    models = {
        "LogisticRegression": (LogisticRegression, [
            {"C": 1.0, "random_state": 42}
        ]),
        "DecisionTree": (DecisionTreeClassifier, [
            {"max_depth": 4, "random_state": 42}
        ]),
        "RandomForest": (RandomForestClassifier, [
            {"n_estimators": 10, "max_depth": 4, "random_state": 42}
        ]),
        "LinearSVC": (SVC, [
            {"kernel": "linear", "C": 1.0, "random_state": 42}
        ]),
        "RBFSVC": (SVC, [
            # Note: Using float for gamma instead of "scale" to make C++ parsing easier
            {"kernel": "rbf", "C": 1.0, "gamma": 0.5, "random_state": 42}
        ]),
        "NaiveBayes": (GaussianNB, [{}]),
        "LDA": (LinearDiscriminantAnalysis, [{}]),
        "QDA": (QuadraticDiscriminantAnalysis, [{}])
    }

    # Start building the C++ header
    cpp_code = "#pragma once\n#include <vector>\n#include <string>\n#include \"../AxML/common.hpp\"\n\n"
    cpp_code += f"inline const int N_TRAIN = {X_train.shape[0]};\n"
    cpp_code += f"inline const int N_GRID = {X_grid.shape[0]};\n\n"

    cpp_code += f"inline const std::vector<AxML::Scalar> X_train = {cpp_float_vec(X_train)};\n"
    cpp_code += f"inline const std::vector<int> y_train = {cpp_int_vec(y_train)};\n\n"
    cpp_code += "// X_grid is flattened row-major. Size: N_GRID * 2\n"
    cpp_code += f"inline const std::vector<AxML::Scalar> X_grid = {cpp_float_vec(X_grid)};\n\n"

    sklearn_preds = {}

    # 4. Train Models, Predict Grid, and Generate C++ Structs
    for name, (ModelClass, configs) in models.items():
        struct_name = f"{name}Config"

        # Build C++ struct fields based on the first config's keys
        fields = ""
        for k, v in configs[0].items():
            cpp_type = "AxML::Scalar" if isinstance(v, float) else "int" if isinstance(v, int) else "std::string"
            fields += f"    {cpp_type} {k};\n"

        cpp_code += f"struct {struct_name} {{\n    std::string test_name;\n{fields}}};\n\n"
        cpp_code += f"inline const std::vector<{struct_name}> {name.lower()}_configs = {{\n"

        for i, config in enumerate(configs):
            # Train Scikit-learn baseline
            model = ModelClass(**config)
            model.fit(X_train, y_train)

            # Since there might be multiple configs per model, we append the index to the name for the npz dict
            config_id = f"{name}_{i}"
            sklearn_preds[config_id] = model.predict(X_grid)

            # Build the C++ struct initializer
            init_vals = [f'"{config_id}"']
            for k in configs[0].keys():
                val = config.get(k, configs[0][k]) # fallback if missing
                init_vals.append(f"{val}f" if isinstance(val, float) else cpp_string(val))

            cpp_code += f"    {{{', '.join(init_vals)}}},\n"

        cpp_code += "};\n\n"

    # 5. Write Outputs
    with open("boundary_data.hpp", "w") as f:
        f.write(cpp_code)

    np.savez("sklearn_boundary_data.npz",
             X_train=X_train, y_train=y_train,
             xx=xx, yy=yy,
             **sklearn_preds)

    print("Generated 'boundary_data.hpp' and 'sklearn_boundary_data.npz'.")

if __name__ == "__main__":
    main()