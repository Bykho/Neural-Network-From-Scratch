#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <cmath>
#include "Matrix.h"

namespace LossFunctions {

    // Compute xentropy loss
    // Predictions: Matrix of predProba (after softmax)
    // targets: Matrix of true labels
    double crossEntropyLoss(const Matrix<double>& predictions,
                            const Matrix<double>& targets);
    
    //Compute xentropy loss for backpropagation
    // Return dL/dy (gradient w.r.t predictions)
    Matrix<double> crossEntropyGradient(const Matrix<double>& predictions,
                                        const Matrix<double>& targets);

    //Helper function to prevent numerical instability.
    //avoid log(0)
    double clip(double x, double epsilon = 1e-7);
};

#endif