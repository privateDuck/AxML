#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "AxML/AxML.hpp"
#include "boundary_data.hpp"
#include "../AxML/discriminant/discriminant_analysis.hpp"
#include "../AxML/nn/mlp.hpp"
#include "AxML/classification/logistic.hpp"
#include "AxML/classification/svc.hpp"

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
    Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    for (const auto&[test_name, max_depth, random_state] : decisiontree_configs) {
        std::cout << "Testing config: " << test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();

        AxML::DecisionTreeParams params;
        params.max_depth = max_depth;
        params.max_features = "all";
        params.random_state = random_state;
        AxML::DecisionTreeClassifier model(params);
        model.fit(X, y);

        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + test_name + ".csv", my_preds);
    }
}

void test_random_forest_classifier() {
    std::cout << "--- Running Random Forest Tests ---\n";

    Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

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

void test_mlp_classifier() {
    std::cout << "--- Running MLP Tests ---\n";
    const Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    AxML::Vector mean = X.colwise().mean();
    AxML::MatrixR centered = X.rowwise() - mean.transpose();

    // Calculate standard deviation (add small epsilon to avoid div by zero)
    AxML::Vector stddev = (centered.array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    AxML::Vector safe_stddev = stddev.cwiseMax(1e-8f);

    // Scale X
    AxML::MatrixR X_scaled = centered.array().rowwise() / safe_stddev.transpose().array();
    AxML::MatrixR X_grid_scaled = (X_grid_mat.rowwise() - mean.transpose()).array().rowwise() / safe_stddev.transpose().array();

    for (const auto& tc : mcpc_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();
        AxML::MLPParams params;
        params.solver = "adam";
        params.activation = tc.activation;
        params.batch_size = 32;
        params.max_iter = 500;
        params.learning_rate = 0.01;
        params.momentum = 0.9;
        params.hidden_layers = tc.hidden_layer_sizes;
        // params.
        AxML::MLPClassifier model(params);
        model.fit(X, y);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

void test_logistic_classifier() {
    std::cout << "--- Running Logistic Tests ---\n";
    const Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    AxML::Vector mean = X.colwise().mean();
    AxML::MatrixR centered = X.rowwise() - mean.transpose();

    // Calculate standard deviation (add small epsilon to avoid div by zero)
    AxML::Vector stddev = (centered.array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    AxML::Vector safe_stddev = stddev.cwiseMax(1e-8f);

    // Scale X
    AxML::MatrixR X_scaled = centered.array().rowwise() / safe_stddev.transpose().array();
    AxML::MatrixR X_grid_scaled = (X_grid_mat.rowwise() - mean.transpose()).array().rowwise() / safe_stddev.transpose().array();

    for (const auto& tc : logisticregression_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();
        AxML::LogisticRegressionParams params;
        params.regularization = "l2";
        AxML::LogisticRegression model(params);
        model.fit(X, y);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

void test_svc_classifier() {
    std::cout << "--- Running svc Tests ---\n";
    const Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    for (const auto& tc : linearsvc_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();
        AxML::LinearSVCParams params;
        params.C = tc.C;
        params.margin = 1.0;
        AxML::LinearSVC model(params);
        model.fit(X, y);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

void test_rbfsvc_classifier() {
    std::cout << "--- Running rbf svc Tests ---\n";
    const Eigen::Map<const AxML::MatrixR> X(X_train.data(), N_TRAIN, 2);
    Eigen::Map<const AxML::VectorI> y(y_train.data(), N_TRAIN);
    Eigen::Map<const AxML::MatrixR> X_grid_mat(X_grid.data(), N_GRID, 2);

    for (const auto& tc : rbfsvc_configs) {
        std::cout << "Testing config: " << tc.test_name << "...\n";

        auto start = std::chrono::high_resolution_clock::now();
        AxML::RBFSVCParams params;
        params.C = tc.C;
        params.gamma = tc.gamma;
        AxML::RBFSVC model(params);
        model.fit(X, y);
        AxML::VectorI my_preds = model.predict(X_grid_mat);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "[COMPLETED] " << tc.test_name << " in " << elapsed.count() << " ms\n";
        write_predictions("cpp_preds_" + tc.test_name + ".csv", my_preds);
    }
}

#include "../MiniDNN/include/MiniDNN.h"

void train_minidnn(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y) {
    using namespace MiniDNN;
    Network net;
    Layer* input = new FullyConnected<ReLU>(400, 100);
    Layer* hidden = new FullyConnected<ReLU>(100, 10);
    Layer* output = new FullyConnected<Identity>(10, 2);
    net.add_layer(std::move(input));
    net.add_layer(std::move(hidden));
    net.add_layer(output);

    net.set_output(new MultiClassEntropy());
    Adam opt;
    net.init();
    net.fit(opt, X, y, 32, 100);
}

void train_our(const AxML::MatrixR& X, const AxML::VectorI& y) {
    AxML::MLPParams params;
    params.hidden_layers = {100, 10};
    params.solver = "adam";
    params.activation = "relu";
    params.batch_size = 32;
    params.max_iter = 100;
    params.learning_rate = 0.001;
    AxML::MLPClassifier model(params);
    model.fit(X, y);
}

void compare_minidnn() {

    Eigen::MatrixXd xm = Eigen::MatrixXd::Random(400, 100);
    const int num_classes = 2;
    const int num_observations = 100;

    // 2. Initialize an all-zero matrix
    Eigen::MatrixXd ym = Eigen::MatrixXd::Zero(num_classes, num_observations);

    // 3. Setup standard C++ random engine for row indices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dis(0, num_classes - 1);

    // 4. Populate each column with exactly one '1'
    for (int col = 0; col < num_observations; ++col) {
        int random_row = dis(gen);
        ym(random_row, col) = 1.0;
    }

    AxML::MatrixR xo = AxML::MatrixR::Random(100, 400);
    AxML::VectorI yo = AxML::VectorI::Random(100);

    auto minidnn_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        train_minidnn(xm, ym);
    }

    auto minidnn_end = std::chrono::high_resolution_clock::now();

    auto our_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        train_our(xo, yo);
    }

    auto our_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_minidnn = minidnn_end - minidnn_start;
    std::chrono::duration<double, std::milli> elapsed_our = our_end - our_start;

    std::cout << "[COMPLETED] " << "MiniDNN in " << elapsed_minidnn.count() << " ms\n";
    std::cout << "[COMPLETED] " << "Ours in " << elapsed_our.count() << " ms\n";
}

int main() {
    test_random_forest_classifier();
    test_decision_tree_classifier();
    test_lda_classifier();
    //test_mlp_classifier();
    test_logistic_classifier();
    //test_svc_classifier();
    //test_rbfsvc_classifier();
    compare_minidnn();
}