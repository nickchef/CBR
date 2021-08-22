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

* image of 32x32 in 3 channels

实现了朴素卷积过程,设图片总像素为n,通道为c, 卷积核边长为m,实际时间复杂度为O(n^2m^2), 空间复杂度为O(n).

实现了简单的OMP多线程, 在i9-9900K上将单图片卷积(8卷积核)+池化+SoftMax过程由6秒降至了2秒.

可以进一步采用的改善措施:
GPU运算, im2col+矩阵运算库(MKL, Eigen, BLAS), 内存排列优化
