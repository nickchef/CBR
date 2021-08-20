//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_CONVLAYER_HPP
#define CBR_CONVLAYER_HPP
#include <Layer.hpp>
#include <chrono>

class ConvLayer: public Layer{

    ConvLayer(int s, int p, int f, int n, int c):stride(s), padding(p), filter_size(f), filter_num(n), channel(c){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine e(seed);
        normal_distribution<float> dis(0,1);

        weight = vector<vector<vector<vector<float>>>>(
                filter_num, vector<vector<vector<float>>>(
                        channel, vector<vector<float>>(
                                filter_size, vector<float>(filter_size,0))));
        for(int i = 0; i < filter_num; i++){
            for(int j = 0; j < channel; j++){
                for(int k = 0; k < filter_size; k++){
                    for(int m = 0; m < filter_size; m++){
                        weight[i][j][k][m] = dis(e);
                    }
                }
            }
        }
    }

    void im2col(Eigen::MatrixXf& target, const vector<vector<vector<float>>>& in){
        for(int i = )
    }

    void forward(const vector<vector<vector<float>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{
        vector<int> output_shape(3,0);
        output_shape[0] = filter_num;
        output_shape[1] = shape[1] + padding - filter_size + 1;
        output_shape[2] = shape[2] + padding - filter_size + 1;

        out = vector<vector<vector<float>>>(output_shape[0], vector<vector<float>>(output_shape[1], vector<float>(output_shape[2], 0)));

//        if(out.size()!=output_shape[0]){
//            out.resize(output_shape[0], vector<vector<double>>(output_shape[1], vector<double>(output_shape[2], 0)));
//        }
//
//        if(out[0].size()!=output_shape[1]){
//            for(vector<vector<double>> i: out){
//                i.resize(output_shape[1], vector<double>(output_shape[2], 0));
//            }
//        }
//
//        if(out[0][0].size()!=output_shape[2]){
//            for(const vector<vector<double>>& i: out){
//                for(vector<double> j: i){
//                    j.resize(output_shape[2], 0);
//                }
//            }
//        }

        if(padding > 0){
            for(vector<vector<float>> layer : in){
                for(vector<float> row : layer){
                    for(int i = 0; i < padding; i++){
                        row.insert(row.end(), 0);
                        row.insert(row.begin(), 0);
                    }
                }
                for(int i = 0; i < padding; i++){
                    layer.insert(layer.begin(), vector<float>(shape[3]+padding*2, 0));
                    layer.insert(layer.end(), vector<float>(shape[3]+padding*2, 0));
                }
            }
        }

        for(int filter = 0; filter < filter_num; filter++){
            for(int channelNo = 0; channelNo < shape[0]; channelNo++){
                for(int y = 0; y <= in[0].size()-filter_size; y+=stride){
                    for(int x = 0; x <= in[0][0].size()-filter; x+=stride){
                        for(int filterx = 0; filterx < filter_size; filterx++){

                        }
                    }
                }
            }
        }

    };

    void forward(vector<vector<vector<BYTE>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{

    };

private:
    int stride;
    int padding;
    int filter_size;
    int filter_num;
    int channel;

    vector<vector<vector<vector<float>>>> weight;

    double bn_weight = 1;
    double bn_bias = 0;
};

#endif //CBR_CONVLAYER_HPP
