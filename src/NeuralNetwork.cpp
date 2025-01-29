#include "NeuralNetwork.h"
#include "Matrix.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include <iostream>
#include <random>

NeuralNetwork::NeuralNetwork(size_t inputDim, size_t outputDim)
    : inputDim(inputDim), outputDim(outputDim) {
    if (inputDim == 0 || outputDim == 0) {
        throw std::invalid_argument("Input and output dimensions must be positive");
    }
}

void NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
    // Add validation here (check dimensions match)
    if (!layer) {
        throw std::invalid_argument("Cannot add null layer");
    }

    if (layers.empty()) {
        if (layer->getInputDim() != inputDim) {
            throw std::invalid_argument("First layer input dimension must match network input dimension");
        }
    } else {
        if (layer->getInputDim() != layers.back()->getOutputDim()) {
            throw std::invalid_argument("Layer input dimension must match previous layer output dimension");
        }
    }
    layers.push_back(layer);
}

Matrix<double> NeuralNetwork::forward(const Matrix<double>& input) const {
    if (layers.empty()) {
        throw std::runtime_error("Cannot forward propagate through empty network");
    }
    // Validate input dimensions
    if (input.getCols() != inputDim) {
        throw std::invalid_argument("Input dimension does not match network input dimension");
    }

    // Forward propagate through each layer
    Matrix<double> current = input;
    for (const auto& layer : layers) {
        current = layer->forward(current);
    }

    return current;
}

void NeuralNetwork::backward(const Matrix<double>& loss_gradient) {
    if (layers.empty()) {
        throw std::runtime_error("No layers in the network for backpropagation.");
    }

    auto gradient = loss_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        try {
            gradient = (*it)->backward(gradient);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error during backpropagation: " + std::string(e.what()));
        }
    }
}


void NeuralNetwork::train(const Matrix<double>& input, const Matrix<double>& targets,
            size_t epochs, double learning_rate) {
    const size_t batch_size = 32;
    const size_t num_samples = input.getRows();
    const size_t num_batches = (num_samples + batch_size - 1) / batch_size;

    try {
        Optimizer optimizer(learning_rate);
        std::cout << "\nStarting training for " << epochs << " epochs..." << std::endl;
        std::cout << "Batch size: " << batch_size << ", Total batches per epoch: " << num_batches << "\n" << std::endl;
        
        std::vector<size_t> indices(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            indices[i] = i;
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            std::shuffle(indices.begin(), indices.end(), gen);

            // Process each batch
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start_idx = batch * batch_size;
                size_t end_idx = std::min(start_idx + batch_size, num_samples);
                size_t current_batch_size = end_idx - start_idx;

                Matrix<double> batch_input(current_batch_size, inputDim);
                Matrix<double> batch_targets(current_batch_size, outputDim);

                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t idx = indices[start_idx + i];
                    for (size_t j = 0; j < inputDim; ++j) {
                        batch_input.at(i, j) = input.at(idx, j);
                    }
                    for (size_t j = 0; j < outputDim; ++j) {
                        batch_targets.at(i, j) = targets.at(idx, j);
                    }
                }

                auto predictions = forward(batch_input);
                double batch_loss = LossFunctions::crossEntropyLoss(predictions, batch_targets);
                epoch_loss += batch_loss;

                auto gradient = LossFunctions::crossEntropyGradient(predictions, batch_targets);
                backward(gradient);
                
                for (auto& layer : layers) {
                    optimizer.update(*layer);
                }
            }

            // Print progress every epoch
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                     << " - Average Loss: " << epoch_loss / num_batches << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Error during training: " + std::string(e.what()));
    }
}


Matrix<double> NeuralNetwork::predict(const Matrix<double>& input) const {
    if (layers.empty()) {
        throw std::runtime_error("Cannot predict with empty network");
    }
    return forward(input);
}


std::shared_ptr<Layer> NeuralNetwork::at(size_t index) const {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers[index];
}