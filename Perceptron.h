#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
class Perceptron {
  private:
    std::vector<double> weights;
    double bias;

  public:
    Perceptron(const std::vector<double>& w0, double b0);

    // activation function
    int Step(double x);
    
    // forward propogation
    int forward(const std::vector<double>& input, int (*activationFunction)(double));

    // getter
    std::vector<double> getWeights();
    double getBias();
};

#endif // Perceptron.h