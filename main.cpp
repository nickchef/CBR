#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <chrono>
#include <NN.hpp>

using namespace std;

int main()
{
    NN nn = NN();
    nn.load();
    nn.train();
}