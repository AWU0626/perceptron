#include "Perceptron.h"

#include <stdexcept>
#include <vector>

// Perceptron constructor
Perceptron::Perceptron(const std::vector<double>& w0, double b0) : weights(w0), bias(b0) {}

// Perceptron's activation function (step function)
int Perceptron::Step(double x) { return 1 ? x >= 0 : 0; }

// forward propogation function
int Perceptron::forward(
  const std::vector<double>& inputs, 
  int (*activationFunction)(double)
) {
  if (inputs.size() != weights.size()) {
    throw std::invalid_argument("Size mismatch. Input size does not match weight size.");
  }

  double dotProduct = 0.0;
  size_t n = weights.size();
  
  for (size_t i = 0; i < n; i++) {
    dotProduct += inputs[i] * weights[i];
  }

  return activationFunction(dotProduct + bias);
}

// getter for weights
std::vector<double> Perceptron::getWeights() { return weights; }

// getter for bias
double Perceptron::getBias() { return bias; }