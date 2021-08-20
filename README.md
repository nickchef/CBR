# CBR
The Convolution, BatchNorm and ReLU is a basic unit of CNN.

This test wants you to write the function of these three parts by C/C++.

* Forward Only; Backward is a PLUS;

* Verify the results by test case and calculate the computation efficiency

Select at least one of below advantage features to implement

* Fuse the Conv + BatchNorm + ReLU into one function to reduce the memory access

    - https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post155s2-file2.pdf

* Parallel with popular parallel technique. e.g. OpenMP, TBB

    - https://computing.llnl.gov/tutorials/openMP/

Using data set cifar-10:

* image of 72x72 in 3 channels

Conv Layer 1:

* filter size 3x3x3
* filter num 10
* stride 1
* padding 0
* Max_Pooling size 2 stride 2

Conv layer 2:

* filter size 3x3x10
* filter num 5
* stride 1
* padding 0
* Max_Pooling size 2 stride 2

2 FC Layer

1 SoftMax Layer