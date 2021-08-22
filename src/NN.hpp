#ifndef CBR_NN_HPP
#define CBR_NN_HPP
#include <ConvLayer.hpp>
#include <FC.hpp>
#include <SoftMax.hpp>
#include <Pooling.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <malloc.h>
#include <memory>

#define CHANNEL (3)
#define IMG_SIZE (32)
#define LABEL_SIZE (1)
#define BATCH_NUM (5)
#define BATCH_SIZE (10000)
#define BATCH_NAME "./cifar-data/data_batch_"
#define BATCH_EXT ".bin"

using namespace std;

class NN {
public:
    explicit NN(bool test=false):test(test){
        // 初始化层及参数, 测试案例中为方便计算仅加入了一个单核卷积层
        if(test){
            layers.push_back(shared_ptr<Layer>(new ConvLayer(1, 1, 3, 1, CHANNEL, true)));
        }else{
            layers.push_back(shared_ptr<Layer>(new ConvLayer(1, 1, 3, 8, CHANNEL)));
            layers.push_back(shared_ptr<Layer>(new Pooling(2, 2)));
            layers.push_back(shared_ptr<Layer>(new SoftMax(10, 17*17*8)));
        }
    }

    void load(){
        //从文件中读取图片信息, 所用数据集为32x32的3通道图片
        vector<string> filename;
        for(int i = 1; i < BATCH_NUM + 1; i ++){
            filename.emplace_back(BATCH_NAME + to_string(i) + BATCH_EXT);
        }

        for(int i = 0; i < BATCH_NUM; i++){
            ifstream file(filename[i], ios::in|ios::binary);
            if(!file){
                cout << "IO FAILURE" << endl;
                return;
            }
            x_batch.emplace_back(vector<vector<vector<vector<BYTE>>>>(BATCH_SIZE));
            y_batch.emplace_back(vector<BYTE>(BATCH_SIZE, 0));
            for(int j = 0; j < BATCH_SIZE; j++){
                file.read((char*)&(y_batch[i][j]), LABEL_SIZE);
                x_batch[i][j] = vector<vector<vector<BYTE>>>(CHANNEL);
                for(int k = 0; k < CHANNEL; k++){
                    x_batch[i][j][k] = vector<vector<BYTE>>(IMG_SIZE);
                    for(int l = 0; l < IMG_SIZE; l++){
                        x_batch[i][j][k][l] = vector<BYTE>(IMG_SIZE);
                        file.read((char*)&x_batch[i][j][k][l][0], IMG_SIZE);
                    }
                }
            }
        }
        
        shape = vector<int>(3, 0);
        shape[0] = CHANNEL;
        shape[1] = IMG_SIZE;
        shape[2] = IMG_SIZE;
    }

    void train(int batch=0){
        auto time = chrono::system_clock::now();
        layers[0]->forward(x_batch[0][0], head, shape);
        int layer_idx = 1;
        for(; layer_idx < layers.size(); layer_idx ++){
            layers[layer_idx]->forward(head, bottom, shape);
            if(++layer_idx<layers.size()){
                layers[layer_idx]->forward(bottom, head, shape);
            }
        }

        auto duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - time);
        cout << "Time cost: " << duration.count() << "ms" << endl;

        cout << "Output shape:";
        for(int i: shape){

            cout << i << " ";
        }
        cout << endl;

        cout << "output" << endl;
        for(const vector<vector<float>>& i : layer_idx%2==0?bottom:head){
            for(const vector<float>& j : i){
                for(float k : j){
                    cout << k << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        if(test){
            assert(head[0][30][0] ==  float(13.758774));
            assert(head[0][30][1] = float(37.0093));
            //计算了矩阵中的部分结果以作为简单正确性的测试
        }
    };

    void loss(){}
    void eval(){}

private:
    vector<vector<vector<vector<vector<BYTE>>>>> x_batch;
    /*
     * 1st Dimension: Batch;
     * 2nd Dimension: Sample;
     * 3rd Dimension: Channel;
     * 4th Dimension: Rows;
     * 5th Dimension: Column
     */
    vector<vector<BYTE>> y_batch;
    /*
     * 1st Dimension: Batch;
     * 2nd Dimension: Label;
     */
    vector<vector<vector<vector<BYTE>>>> x_test;
    vector<BYTE> y_test;

    vector<shared_ptr<Layer>> layers;

    vector<vector<vector<float>>> head;
    vector<vector<vector<float>>> bottom;

    vector<int> shape;

    bool test;
    float lr = 0.01;
};



#endif //CBR_NN_HPP
