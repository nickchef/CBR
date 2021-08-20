//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_SOFTMAX_HPP
#define CBR_SOFTMAX_HPP
class SoftMax: public Layer{
public:
    SoftMax(int class_num, int feature_num):neuron_num(class_num), input_feature_num(feature_num){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine e(seed);
        normal_distribution<float> dis(0,1);

        weight = vector<vector<float>>(class_num, vector<float>(feature_num,0));
        bias = vector<float>(class_num, 0);
        for(vector<float> &neuron: weight){
            for(float &_weight: neuron){
                _weight = (float)(dis(e)*0.1);
            }
        }
        for(float &_bias: bias){
            _bias = (float)(dis(e))*0.1);
        }
    }

    void forward(vector<vector<vector<float>>>& in, vector<vector<vector<float>>>& out, vector<int>& shape) override{
        out = vector<vector<vector<float>>>(1, vector<vector<float>>(1, vector<float>(neuron_num, 0)));

        for(int neuron = 0; neuron < neuron_num; neuron++){
            int cursor = 0;
            for(const vector<vector<float>>& i:in){
                for(const vector<float>&j:i){
                    for(float k:j){
                        out[0][0][neuron] += k*weight[neuron][cursor++];
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

    void backward() override{

    }

private:
    vector<vector<float>> weight;
    vector<float> bias;

    int input_feature_num;
    int neuron_num;
};
#endif //CBR_SOFTMAX_HPP
