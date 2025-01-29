#include "Optimizer.h"
#include "Layer.h"


Optimizer::Optimizer(double lr): learning_rate(lr) {
    if (lr <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
};

void Optimizer::update(Layer& layer) {
    layer.updateParameters(learning_rate);
}