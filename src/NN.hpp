#ifndef CBR_NN_HPP
#define CBR_NN_HPP
#include <ConvLayer.hpp>
#include <FC.hpp>
#include <SoftMax.hpp>
#include <Pooling.hpp>
#include <Dtype.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <array>
#include <omp.h>
#include <cstring>
#include <malloc.h>
#include <memory>

#define CHANNEL (3)
#define PIXEL_NUM (3072)
#define IMG_SIZE (32)
#define LABEL_SIZE (1)
#define BATCH_NUM (5)
#define BATCH_SIZE (10000)
#define BATCH_NAME "./cifar-data/data_batch_"
#define BATCH_EXT ".bin"

using namespace std;

class NN {
public:

    NN(){
        layers.push_back(shared_ptr<Layer>(new ConvLayer(1, 2 , 3, 8, 3)));
    };

    void init(){

    }

    void load(){
        vector<string> filename;
        for(int i = 1; i < BATCH_NUM + 1; i ++){
            filename.emplace_back(BATCH_NAME + to_string(i) + BATCH_EXT);
            cout << filename[i-1] << endl;
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
        layers[0]->forward(x_batch[0][0], bottom, shape);
        for(int i: shape){
            cout << i << endl;
        }

        for(const vector<vector<float>>& i : bottom){
            for(const vector<float>& j : i){
                for(float k : j){
                    cout << k << " ";
                }
                cout << endl;
            }
        }
    };

    void loss();
    void eval();

private:
    vector<vector<vector<vector<vector<BYTE>>>>> x_batch;
    /*
     * 1st Dimension: Batch;
     * 2nd Dimension: Sample;
     * 3rd Dimension: img;
     */
    vector<vector<BYTE>> y_batch;
    /*
     * 1st Dimension: Batch;
     * 2nd Dimension: Label;
     */
    vector<vector<vector<vector<BYTE>>>> x_test;
    /*
     * 1st Dimension: Sample;
     * 2nd Dimension: img;
     */
    vector<BYTE> y_test;

    vector<shared_ptr<Layer>> layers;

    vector<vector<vector<float>>> head;
    vector<vector<vector<float>>> bottom;

    vector<int> shape;

    float lr = 0.01;
};



#endif //CBR_NN_HPP
