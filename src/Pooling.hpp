//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_POOLING_HPP
#define CBR_POOLING_HPP
#include <Layer.hpp>

class Pooling:public Layer{
public:
    explicit Pooling(int ps = 2, int s = 2):pool_size(ps), stride(s){}

    void forward(vector<vector<vector<float>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{
        out = vector<vector<vector<float>>>(shape[0], vector<vector<float>>(shape[1]/2, vector<float>(shape[2]/2, 0)));
        #pragma omp parallel for collapse(5)
        for(int channel = 0; channel < shape[0]; channel++){
            for(int y = 0; y <= shape[1] - pool_size; y+=stride){
                for(int x = 0; x <= shape[2] - pool_size; x+=stride){
                    for(int xs = 0; xs < pool_size; xs++){
                        for(int ys = 0; ys < pool_size; ys++){
                            out[channel][y/pool_size][x/pool_size] = in[channel][y+ys][x+xs] >= out[channel][y/pool_size][x/pool_size]?
                                    in[channel][y+ys][x+xs] : out[channel][y/pool_size][x/pool_size];
                        }
                    }
                }
            }
        }
        shape[1] = shape[1]/2;
        shape[2] = shape[2]/2;
    }

private:
    int stride;
    int pool_size;

};
#endif //CBR_POOLING_HPP
