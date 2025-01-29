#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include "ActivationFunctions.h"

class Layer {
protected:
    size_t inputDim;
    size_t outputDim;
    ActivationFunctions::ActivationFunctionPair activationPair;
    
    // Stored for backward pass
    Matrix<double> lastInput;
    Matrix<double> lastOutput;
    Matrix<double> lastActivation;
    
    // Gradients
    Matrix<double> weightsGrad;
    Matrix<double> biasGrad;

public:
    Layer(size_t inputDim, size_t outputDim, 
          ActivationFunctions::ActivationFunctionPair activationPair)
        : inputDim(inputDim), 
          outputDim(outputDim), 
          activationPair(activationPair),
          lastInput(1, inputDim),
          lastOutput(1, outputDim),
          lastActivation(1, outputDim),
          weightsGrad(outputDim, inputDim),
          biasGrad(1, outputDim) {}
    
    virtual ~Layer() = default;

    // Core layer operations
    virtual Matrix<double> forward(const Matrix<double>& input) = 0;
    virtual Matrix<double> backward(const Matrix<double>& outputGradient) = 0;
    virtual void updateParameters(double learningRate) = 0;

    // Getters for dimensions
    size_t getInputDim() const { return inputDim; }
    size_t getOutputDim() const { return outputDim; }

    // Gradient access
    const Matrix<double>& getWeightsGrad() const { return weightsGrad; }
    const Matrix<double>& getBiasGrad() const { return biasGrad; }
    
    // Reset gradients between batches
    virtual void resetGradients() = 0;
};

#endif