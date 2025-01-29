#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>

namespace ActivationFunctions {
   // Type definition for activation function pointers
   using ActivationFunction = double(*)(double);
   using ActivationDerivative = double(*)(double);

   // Function pairs struct to keep functions with their derivatives
   struct ActivationFunctionPair {
       ActivationFunction activation;
       ActivationDerivative derivative;
   };

   // Function declarations
   double relu(double x);
   double reluDerivative(double x);

   double sigmoid(double x);
   double sigmoidDerivative(double x);

   double tanh(double x);
   double tanhDerivative(double x);

   // Get function pairs
   ActivationFunctionPair getRelu();
   ActivationFunctionPair getSigmoid();
   ActivationFunctionPair getTanh();
}

#endif