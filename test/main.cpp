#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "AxML/AxML.hpp"
#include "boundary_data.hpp"
#include "AxML/discriminant_analysis.hpp"

static const std::filesystem::path TEST_DIR = AXML_TEST_DIR;

bool assert_float_vector_equal(const std::vector<float>& expected, const std::vector<float>& actual, float tol = 1e-4f) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch: Expected " << expected.size() << ", got " << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": Expected " << expected[i] << ", got " << actual[i] << "\n";
            return false;
        }
    }
    return true;
}

bool assert_probas_add_to_one(const AxML::MatrixR& probas, float tol = 1e-4f) {
    for (int i = 0; i < probas.rows(); ++i) {
        AxML::Scalar sum = probas.row(i).sum();
        if (std::abs(sum - 1.0f) > tol) {
            std::cerr << "Probas do not sum to 1 at row " << i << ": sum = " << sum << "\n";
            return false;
        }
    }
    return true;
}

// Utility for integer vectors
bool assert_int_vector_equal(const std::vector<int>& expected, const AxML::VectorI& actual) {
    if (expected.size() != actual.size()) return false;
    int sum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual(i)) {
            std::cout << expected[i] << " != " << actual[i] << " ,at i: " << i << "\n";
            sum++;
        }
    }
    if (sum != 0) {
        std::cout << "\n[FAIL] " << static_cast<float>(sum)/static_cast<float>(expected.size()) * 100 << "% Inconsistent \n" << std::endl;
        return false;
    }
    return true;
}

void write_predictions(const std::string& filename, const std::span<const int> preds) {
    std::ofstream out(TEST_DIR / "data" / filename);
    for (size_t i = 0; i < preds.size(); ++i) {
        out << preds[i] << (i == preds.size() - 1 ? "" : ",");
    }
    out.close();
}

void test_decision_tree_classifier() {
    std::cout << "--- Running Decision Tree Tests ---\n";

    for (const auto&[test_name, max_depth, random_state] : decisiontree_configs) {
        std::cout << "Testing config: " << test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();

        AxML::DecisionTreeParams params;
        params.max_depth = max_depth;
        params.max_features = "all";
        params.random_state = random_state;
        AxML::DecisionTreeClassifier model(params);
        Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
        Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
        model.fit(X, y);

        Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + test_name + ".csv", my_preds);
    }
}

void test_random_forest_classifier() {
    std::cout << "--- Running Random Forest Tests ---\n";

    for (const auto& tc : randomforest_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();

        AxML::RandomForestParams params;
        params.max_depth = tc.max_depth;
        params.n_estimators = tc.n_estimators;
        params.max_features = "all";
        params.random_state = tc.random_state;
        params.n_threads = 1;
        AxML::RandomForestClassifier model(params);
        Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
        Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
        Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);
        model.fit(X, y);

        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

void test_lda_classifier() {
    std::cout << "--- Running LDA Tests ---\n";
    const Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    for (const auto& tc : lda_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();

        AxML::LinearDiscriminantAnalysis model;
        model.fit(X, y);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

int main() {
    test_decision_tree_classifier();
    test_random_forest_classifier();
    test_lda_classifier();
}