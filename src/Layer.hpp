//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_LAYER_H
#define CBR_LAYER_H
#include <Eigen/Core>
#include <vector>
#include <random>
#include <numeric>
#include <chrono>

using namespace std;

typedef unsigned char BYTE;

class Layer {
public:
    virtual void forward(vector<vector<vector<float>>>&, vector<vector<vector<float>>>&, vector<int>&){};
    virtual void forward(vector<vector<vector<BYTE>>>&, vector<vector<vector<float>>>&, vector<int>&){};
    virtual void backward(){};
};



#endif //CBR_LAYER_H