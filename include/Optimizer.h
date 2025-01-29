#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"

class Optimizer {
private:
    double learning_rate;

public:
    Optimizer(double lr);

    void update(Layer& inputLayer);
};

#endif