#include <iostream>
#include <random>
#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace std;

int main()
{
    default_random_engine e;
    normal_distribution<float> u(0, 1);
    Eigen::MatrixXf a = (Eigen::MatrixXf::Random(5,5)).array().abs();

    cout << a << endl;
}