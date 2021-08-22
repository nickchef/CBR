#ifndef CBR_CONVLAYER_HPP
#define CBR_CONVLAYER_HPP
#include <Layer.hpp>
#include <iostream>

class ConvLayer: public Layer{
public:
    ConvLayer(int s, int p, int f, int n, int c, bool test=false):stride(s), padding(p), filter_size(f), filter_num(n), channel(c){
        //初始化权重, 测试下使用了固定的整数权重, 非测试时为正态分布随机初始化
        weight = vector<vector<vector<vector<float>>>>(
                filter_num, vector<vector<vector<float>>>(
                        channel, vector<vector<float>>(
                                filter_size, vector<float>(filter_size,0))));
        if(test){
            #pragma omp parallel for collapse(4)
            for(int i = 0; i < filter_num; i++){
                for(int j = 0; j < channel; j++){
                    for(int k = 0; k < filter_size; k++){
                        for(int m = 0; m < filter_size; m++){
                            weight[i][j][k][m] = (float)(m+k);
                        }
                    }
                }
            }
        }else{
            unsigned seed = chrono::system_clock::now().time_since_epoch().count();
            default_random_engine e(seed);
            normal_distribution<float> dis(0,1);

            //        int weightMat_row = f*f*c;
            //        weightMat = MatrixXf(weightMat_row, f);
            //        for(int i = 0; i < f; i ++){
            //            for(int j = 0; j < weightMat_col; j++){
            //                weightMat(i, j) = dis(e);
            //            }
            //        }
            //

            #pragma omp parallel for collapse(4)
            for(int i = 0; i < filter_num; i++){
                for(int j = 0; j < channel; j++){
                    for(int k = 0; k < filter_size; k++){
                        for(int m = 0; m < filter_size; m++){
                            weight[i][j][k][m] = (float)(dis(e) * 0.01);
                        }
                    }
                }
            }
        }
    }

//    void im2col(Eigen::MatrixXf& target, const vector<vector<vector<float>>>& in){
//        for(int filter = 0; filter < filter_num; filter++){
//            for(int channelNo = 0; channelNo < shape[0]; channelNo++){
//                for(int y = 0; y <= in[0].size()-filter_size; y+=stride){
//                    for(int x = 0; x <= in[0][0].size()-filter; x+=stride){
//                        for(int filterx = 0; filterx < filter_size; filterx++){
//
//                        }
//                    }
//                }
//            }
//        }
//    }

    void forward(vector<vector<vector<BYTE>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{
        vector<int> output_shape(3,0);
        //输出尺寸计算
        output_shape[0] = filter_num;
        output_shape[1] = shape[1] + padding*2 - filter_size + 1;
        output_shape[2] = shape[2] + padding*2 - filter_size + 1;

        out = vector<vector<vector<float>>>(output_shape[0], vector<vector<float>>(output_shape[1], vector<float>(output_shape[2], 0)));
        //padding过程
        if(padding > 0){
            for(vector<vector<BYTE>>& layer : in){
                for(vector<BYTE>& row : layer){
                    for(int i = 0; i < padding; i++){
                        row.insert(row.end(), 0);
                        row.insert(row.begin(), 0);
                    }
                }
                for(int i = 0; i < padding; i++){
                    layer.insert(layer.begin(), vector<BYTE>(shape[2]+padding*2, 0));
                    layer.insert(layer.end(), vector<BYTE>(shape[2]+padding*2, 0));
                }
            }
        }

        //朴素卷积过程
        #pragma omp parallel for collapse(6)
        for(int filter = 0; filter < filter_num; filter++){
            for(int x = 0; x < output_shape[2]; x+=stride){
                for(int y = 0; y < output_shape[1]; y+=stride){
                    for(int filterx = 0; filterx < filter_size; filterx++){
                        for(int filtery = 0; filtery < filter_size; filtery++){
                            for(int channelNo = 0; channelNo < shape[0]; channelNo++){
                                #pragma omp atomic
                                out[filter][y][x] += weight[filter][channelNo][filtery][filterx] * (float)in[channelNo][y+filtery][x+filterx];
                            }
                        }
                    }
                }
            }
        }
        //Bn和relu过程
        #pragma omp parallel for
        for(int filter = 0; filter < filter_num; filter++){
            float sum = 0;
            for(int y = 0; y < output_shape[1]; y++){
                sum = accumulate(out[filter][y].begin(), out[filter][y].end(), sum);
            }
            float mean = sum/(float)(output_shape[1]*output_shape[2]);
            float accum = 0;

            for(int y = 0; y < output_shape[1]; y++){
                for(int x = 0; x < output_shape[2]; x++){
                    accum += (out[filter][y][x]-mean)*(out[filter][y][x]-mean);
                }
            }

            float stdev = sqrt(accum/(float)(output_shape[1]*output_shape[2]));

            for(vector<float>& row: out[filter]){
                for(float& i: row){
                    i = (float)(((i-mean) / sqrt(stdev))*bn_weight + bn_bias);//bn
                    i = i > 0? i : 0;//ReLU
                }
            }
        }

        shape = output_shape;
    };

    void backward() override{};

private:
    int stride;
    int padding;
    int filter_size;
    int filter_num;
    int channel;

    vector<vector<vector<vector<float>>>> weight;
//    Eigen::MatrixXf weightMat;

    double bn_weight = 1;
    double bn_bias = 0;
};

#endif //CBR_CONVLAYER_HPP
