//
// Created by 0 on 2021/8/19.
//

#ifndef CBR_POOLING_HPP
#define CBR_POOLING_HPP

class Pooling:public Layer{
public:
    explicit Pooling(int ps = 2, int s = 2):pool_size(ps), stride(s){}

    void forward() {

    }

    void backward() override {

    }

private:
    int stride;
    int pool_size;

};
#endif //CBR_POOLING_HPP
