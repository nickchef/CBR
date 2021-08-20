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

8/21 Update:
Basically finished forward prop. Many optimization need to be done. Im2Col is so hard, MKL is so hard.