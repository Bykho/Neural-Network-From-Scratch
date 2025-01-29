#include "DenseLayer.h"
#include "Matrix.h"
#include <fstream>
#include <iostream>

DenseLayer::DenseLayer(size_t inputDim, size_t outputDim,
            ActivationFunctions::ActivationFunctionPair activationPair )
    : Layer(inputDim, outputDim, activationPair),
        weights(outputDim, inputDim),
        biases(1, outputDim)
{
    initializeParameters();
}

Matrix<double> DenseLayer::forward(const Matrix<double>& input) {
    // Store input for backprop
    lastInput = input;
    lastActivation = input * weights.transpose();
    
    // Add biases to each row
    for(size_t i = 0; i < lastActivation.getRows(); i++) {
        for(size_t j = 0; j < lastActivation.getCols(); j++) {
            lastActivation.at(i,j) += biases.at(0,j);
        }
    }

    lastOutput = lastActivation.apply(activationPair.activation);
    return lastOutput;
}

// The backpropagation algorithm works by computing partial derivatives through the chain rule.
// We need three key gradients here:
// 1. How the loss changes with respect to the layer's output (passed in)
// 2. How the output changes with respect to the weights
// 3. How the output changes with respect to the inputs (for the next layer)
Matrix<double> DenseLayer::backward(const Matrix<double>& outputGradient) {
    Matrix<double> dy_dz = lastActivation.apply(activationPair.derivative);
    Matrix<double> dL_dz = outputGradient.hadamard(dy_dz);
    
    // Computing weight gradients requires a transpose to match dimensions
    Matrix<double> dL_dz_t = dL_dz.transpose();
    Matrix<double> weight_update = dL_dz_t * lastInput;
    weightsGrad = weightsGrad + weight_update;
    
    // Sum gradients across batch for biases
    for(size_t i = 0; i < outputDim; i++) {
        double sum = 0.0;
        for(size_t j = 0; j < dL_dz.getRows(); j++) {
            sum += dL_dz.at(j, i);
        }
        biasGrad.at(0, i) = biasGrad.at(0, i) + sum;
    }
    
    Matrix<double> dL_dx = dL_dz * weights;
    return dL_dx;
}

void DenseLayer::updateParameters(double learningRate) {
    // Standard gradient descent update
    weights = weights - weightsGrad * learningRate;
    biases = biases - biasGrad * learningRate;
    resetGradients();
}

void DenseLayer::resetGradients() {
    // Zero out gradients between batches
    weightsGrad = Matrix<double>(outputDim, inputDim);
    biasGrad = Matrix<double>(1, outputDim);
}

void DenseLayer::initializeParameters() {
    // The scale factor sqrt(2/n) prevents early layers from having vanishingly small gradients.
    double scale = std::sqrt(2.0 / inputDim);
    weights.randomize(0.0, scale);
    biases = Matrix<double>(1, outputDim, 0.0);
}

void DenseLayer::saveWeights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Need to save dimensions for loading later
    size_t w_rows = weights.getRows();
    size_t w_cols = weights.getCols();
    size_t b_size = biases.getCols();

    file.write(reinterpret_cast<const char*>(&w_rows), sizeof(w_rows));
    file.write(reinterpret_cast<const char*>(&w_cols), sizeof(w_cols));
    file.write(reinterpret_cast<const char*>(&b_size), sizeof(b_size));

    // Write raw bytes
    for (size_t i = 0; i < w_rows; ++i) {
        for (size_t j = 0; j < w_cols; ++j) {
            double val = weights.at(i, j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }

    for (size_t i = 0; i < b_size; ++i) {
        double val = biases.at(0, i);
        file.write(reinterpret_cast<const char*>(&val), sizeof(double));
    }
}

void DenseLayer::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    size_t w_rows, w_cols, b_size;
    file.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));
    file.read(reinterpret_cast<char*>(&w_cols), sizeof(w_cols));
    file.read(reinterpret_cast<char*>(&b_size), sizeof(b_size));

    if (w_rows != weights.getRows() || w_cols != weights.getCols() || 
        b_size != biases.getCols()) {
        throw std::runtime_error("Loaded weight dimensions do not match layer dimensions");
    }

    for (size_t i = 0; i < w_rows; ++i) {
        for (size_t j = 0; j < w_cols; ++j) {
            double val;
            file.read(reinterpret_cast<char*>(&val), sizeof(double));
            weights.at(i, j) = val;
        }
    }

    for (size_t i = 0; i < b_size; ++i) {
        double val;
        file.read(reinterpret_cast<char*>(&val), sizeof(double));
        biases.at(0, i) = val;
    }
}