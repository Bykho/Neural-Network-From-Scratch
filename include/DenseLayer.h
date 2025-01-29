#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include <string>

class DenseLayer : public Layer {
private:
    Matrix<double> weights;
    Matrix<double> biases;
    Matrix<double> computePreActivation(const Matrix<double>& input);

public:
    DenseLayer(size_t inputDim, size_t outputDim,
            ActivationFunctions::ActivationFunctionPair activationPair);

    //Must implement all virtual methods from Layer.h
    virtual Matrix<double> forward(const Matrix<double>& input) override;
    virtual Matrix<double> backward(const Matrix<double>& outputGradient) override;
    virtual void updateParameters(double learningRate) override;
    virtual void resetGradients() override;

    const Matrix<double>& getWeights() const { return weights; }
    const Matrix<double>& getBiases() const { return biases; }

    void initializeParameters();

    void saveWeights(const std::string& filename) const;
    void loadWeights(const std::string& filename);

};

#endif