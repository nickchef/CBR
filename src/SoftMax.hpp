#ifndef CBR_SOFTMAX_HPP
#define CBR_SOFTMAX_HPP
#include <Layer.hpp>

class SoftMax: public Layer{
public:
    SoftMax(int class_num, int feature_num):neuron_num(class_num){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine e(seed);
        normal_distribution<float> dis(0,1);

        weight = vector<vector<float>>(class_num, vector<float>(feature_num,0));
        bias = vector<float>(class_num, 0);

        #pragma omp parallel for collapse(2)
        for(int neuron = 0; neuron < class_num; neuron++){
            for(int feature_weight = 0; feature_weight < feature_num; feature_weight++){
                weight[neuron][feature_weight] = (float)(dis(e)*0.01);
            }
        }
        for(int neuron=0; neuron < class_num; neuron++){
            bias[neuron] = (float)(dis(e)*0.01);
        }
    }

    void forward(vector<vector<vector<float>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{
        out = vector<vector<vector<float>>>(1, vector<vector<float>>(1, vector<float>(neuron_num, 0)));

        for(int neuron = 0; neuron < neuron_num; neuron++){
            int cursor = 0;
            #pragma omp parallel for collapse(3)
            for(int channel = 0; channel < shape[0]; channel++){
                for(int y = 0; y < shape[1]; y++){
                    for(int x = 0; x < shape[2]; x++){
                        #pragma omp atomic
                        out[0][0][neuron] += in[channel][y][x] * weight[neuron][cursor++];
                    }
                }
            }
            out[0][0][neuron] += bias[neuron];
        }

        float base = 0;

        for(float &i: out[0][0]){
            i = exp(i);
            base += i;
        }
        for(float &i: out[0][0]){
            i = i/base;
        }
    }

    void backward() override{}

private:
    vector<vector<float>> weight;
    vector<float> bias;

    int neuron_num;
};
#endif //CBR_SOFTMAX_HPP
