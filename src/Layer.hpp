//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_LAYER_H
#define CBR_LAYER_H
#include <Math.hpp>
#include <Eigen/Core>
#include <vector>
#include <Dtype.hpp>
#include <random>

using namespace std;

typedef unsigned char BYTE;

class Layer {
public:
    virtual void forward(vector<vector<vector<float>>>&, vector<vector<vector<float>>>&, vector<int>&) = 0;
    virtual void forward(vector<vector<vector<BYTE>>>&, vector<vector<vector<float>>>&, vector<int>&) = 0;
    virtual void backward() = 0;
};


#endif //CBR_LAYER_H