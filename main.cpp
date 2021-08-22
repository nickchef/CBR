#include <NN.hpp>

using namespace std;

int main(int argc, char** argv){
    NN nn = NN(true);
    nn.load();
    nn.train();
}