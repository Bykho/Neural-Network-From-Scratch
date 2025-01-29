#include "LossFunction.h"
#include <cmath>

namespace LossFunctions {
    double crossEntropyLoss(const Matrix<double>& predictions, const Matrix<double>& targets) {
        if (predictions.getRows() != targets.getRows() || 
            predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets dimensions must match");
        }
        
        double totalLoss = 0.0;
        for (size_t i = 0; i < predictions.getRows(); i++) {
            for (size_t j = 0; j < predictions.getCols(); j++) {
                // Only compute for non-zero targets (one-hot encoded)
                if (targets.at(i,j) > 0) {
                    totalLoss += targets.at(i,j) * std::log(clip(predictions.at(i,j)));
                }
            }
        }
        
        return -totalLoss / predictions.getRows();
    }

    Matrix<double> crossEntropyGradient(const Matrix<double>& predictions, const Matrix<double>& targets) {
        if (predictions.getRows() != targets.getRows() || 
            predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets dimensions must match");
        }
        // Simply return predictions - targets
        return predictions - targets;
    }

    double clip(double x, double epsilon) {
        // Clip value between epsilon and 1-epsilon
        return std::max(epsilon, std::min(1.0 - epsilon, x));
    }

}