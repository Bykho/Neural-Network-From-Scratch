#include "MNISTLoader.h"
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "ActivationFunctions.h"
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

// Normalize pixel values to [0,1]
Matrix<double> createInputMatrix(const std::vector<std::vector<double>>& mnist_data) {
    if (mnist_data.empty()) return Matrix<double>(0, 0);
    Matrix<double> input(mnist_data.size(), mnist_data[0].size());
    for (size_t i = 0; i < mnist_data.size(); i++) {
        for (size_t j = 0; j < mnist_data[0].size(); j++) {
            input.at(i, j) = mnist_data[i][j] / 255.0;
        }
    }
    return input;
}

/* One-hot encoding converts integer labels to binary vectors.
   Each label becomes a vector of zeros with a single 1 at the index
   corresponding to the digit class. This representation helps with
   training stability and loss computation. */
Matrix<double> createTargetMatrix(const std::vector<uint8_t>& labels, size_t num_classes = 10) {
    Matrix<double> target(labels.size(), num_classes, 0.0);
    for (size_t i = 0; i < labels.size(); i++) {
        target.at(i, labels[i]) = 1.0;
    }
    return target;
}

struct Metrics {
    double accuracy;
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1_score;
    std::vector<std::vector<int>> confusion_matrix;
};

/* The confusion matrix reveals model behavior across all classes.
   Each row represents actual labels, columns are predictions.
   Diagonal elements show correct classifications.
   Off-diagonal elements expose where the model makes mistakes. */
Metrics calculateMetrics(const Matrix<double>& predictions, const std::vector<uint8_t>& true_labels) {
    Metrics metrics;
    size_t num_classes = 10;
    size_t num_samples = predictions.getRows();
    
    metrics.confusion_matrix = std::vector<std::vector<int>>(num_classes, std::vector<int>(num_classes, 0));
    metrics.precision = std::vector<double>(num_classes, 0.0);
    metrics.recall = std::vector<double>(num_classes, 0.0);
    metrics.f1_score = std::vector<double>(num_classes, 0.0);
    
    int correct = 0;
    
    // Track predictions vs ground truth
    for (size_t i = 0; i < num_samples; i++) {
        int predicted_class = 0;
        double max_prob = predictions.at(i, 0);
        for (size_t j = 1; j < num_classes; j++) {
            if (predictions.at(i, j) > max_prob) {
                max_prob = predictions.at(i, j);
                predicted_class = j;
            }
        }
        
        int true_class = true_labels[i];
        metrics.confusion_matrix[true_class][predicted_class]++;
        
        if (predicted_class == true_class) {
            correct++;
        }
    }
    
    // Per-class metrics using true/false positives/negatives
    for (size_t i = 0; i < num_classes; i++) {
        int tp = metrics.confusion_matrix[i][i];
        int fp = 0, fn = 0;
        
        for (size_t j = 0; j < num_classes; j++) {
            if (j != i) {
                fp += metrics.confusion_matrix[j][i];
                fn += metrics.confusion_matrix[i][j];
            }
        }
        
        // Small epsilon prevents division by zero
        metrics.precision[i] = tp / (tp + fp + 1e-10);
        metrics.recall[i] = tp / (tp + fn + 1e-10);
        metrics.f1_score[i] = 2 * metrics.precision[i] * metrics.recall[i] / 
                             (metrics.precision[i] + metrics.recall[i] + 1e-10);
    }
    
    metrics.accuracy = static_cast<double>(correct) / num_samples;
    return metrics;
}

void printMetrics(const Metrics& metrics) {
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(4) 
              << metrics.accuracy * 100 << "%" << std::endl;
    
    std::cout << "\nPer-class metrics:\n";
    std::cout << "Digit\tPrecision\tRecall\t\tF1 Score\n";
    std::cout << "------------------------------------------------\n";
    for (size_t i = 0; i < 10; i++) {
        std::cout << i << "\t" 
                  << std::fixed << std::setprecision(4) << metrics.precision[i] * 100 << "%\t\t"
                  << std::fixed << std::setprecision(4) << metrics.recall[i] * 100 << "%\t\t"
                  << std::fixed << std::setprecision(4) << metrics.f1_score[i] * 100 << "%\n";
    }
    
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\n";
    std::cout << "   ";
    for (size_t i = 0; i < 10; i++) std::cout << std::setw(6) << i;
    std::cout << "\n";
    
    for (size_t i = 0; i < 10; i++) {
        std::cout << i << "  ";
        for (size_t j = 0; j < 10; j++) {
            std::cout << std::setw(6) << metrics.confusion_matrix[i][j];
        }
        std::cout << "\n";
    }
}

int main() {
    MNISTLoader loader;
    std::cout << "Loading MNIST data..." << std::endl;
    
    if (!loader.verifyFiles("data/train-images-idx3-ubyte", 
                           "data/train-labels-idx1-ubyte")) {
        std::cerr << "Training data files invalid!" << std::endl;
        return 1;
    }

    try {
        // Architecture
        NeuralNetwork network(784, 10);
        network.addLayer(std::make_shared<DenseLayer>(784, 256, ActivationFunctions::getRelu()));
        network.addLayer(std::make_shared<DenseLayer>(256, 128, ActivationFunctions::getRelu()));
        network.addLayer(std::make_shared<DenseLayer>(128, 64, ActivationFunctions::getRelu()));
        network.addLayer(std::make_shared<DenseLayer>(64, 10, ActivationFunctions::getSigmoid()));

        // Load training data
        std::vector<std::vector<double>> train_images = loader.loadImages("data/train-images-idx3-ubyte");
        std::vector<uint8_t> train_labels = loader.loadLabels("data/train-labels-idx1-ubyte");
        
        Matrix<double> train_data = createInputMatrix(train_images);
        Matrix<double> train_targets = createTargetMatrix(train_labels);

        // 10 epochs, learning rate 0.01
        network.train(train_data, train_targets, 10, 0.01);

        Matrix<double> train_predictions = network.predict(train_data);
        std::cout << "\nTraining Set Metrics:" << std::endl;
        Metrics train_metrics = calculateMetrics(train_predictions, train_labels);
        printMetrics(train_metrics);
        
        // Evaluate generalization on test set
        if (loader.verifyFiles("data/t10k-images-idx3-ubyte", 
                              "data/t10k-labels-idx1-ubyte")) {
            std::vector<std::vector<double>> test_images = loader.loadImages("data/t10k-images-idx3-ubyte");
            std::vector<uint8_t> test_labels = loader.loadLabels("data/t10k-labels-idx1-ubyte");
            
            Matrix<double> test_data = createInputMatrix(test_images);
            Matrix<double> test_predictions = network.predict(test_data);
            
            std::cout << "\nTest Set Metrics:" << std::endl;
            Metrics test_metrics = calculateMetrics(test_predictions, test_labels);
            printMetrics(test_metrics);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}