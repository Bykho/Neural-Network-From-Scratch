#include "ActivationFunctions.h"
#include <cmath>

namespace ActivationFunctions {
    double relu(double x) {
        if (x > 0.0) {
            return x;
        }
        return 0.0;
    }

    double reluDerivative(double x) {
        if (x > 0.0) {
            return 1.0;
        }
        return 0.0;
    }

    double sigmoid(double x) {
        return 1.0/(1.0+std::exp(-x));
    }

    double sigmoidDerivative(double x) {
        return sigmoid(x) * (1.0- sigmoid(x));
    }

    double tanh(double x) {
        return std::tanh(x);
    }

    double tanhDerivative(double x) {
        return 1 - std::pow(std::tanh(x), 2);
    }

    ActivationFunctionPair getRelu() {
        return {relu, reluDerivative};
    }

    ActivationFunctionPair getSigmoid() {
        return {sigmoid, sigmoidDerivative};
    }

    ActivationFunctionPair getTanh() {
        return {tanh, tanhDerivative};
    }
}

