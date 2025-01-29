#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "DenseLayer.h"
#include "Optimizer.h"
#include "LossFunction.h"
#include "Matrix.h"
#include <vector>
#include <memory>

class NeuralNetwork {
private:
    std::vector<std::shared_ptr<Layer>> layers;
    size_t inputDim;
    size_t outputDim;
    
public:
    // Constructor
    NeuralNetwork(size_t inputDim, size_t outputDim);
    
    void addLayer(std::shared_ptr<Layer> layer);
    
    Matrix<double> forward(const Matrix<double>& input) const;
    void backward(const Matrix<double>& loss_gradient);
    
    void train(const Matrix<double>& input, const Matrix<double>& targets,
               size_t epochs, double learning_rate);
    Matrix<double> predict(const Matrix<double>& input) const;
    
    std::shared_ptr<Layer> at(size_t index) const;
    size_t getLayerCount() const { return layers.size(); }
    void clear() { layers.clear(); }
};

#endif
