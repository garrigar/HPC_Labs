# Lab #2 - Genetic Algorithm
## Rail Gabbasov, 6132-010402D

## Description
In this work, an implementation of the genetic algorithm both on GPU and CPU is done.

The GPU version uses `numba.cuda` and `torch` libraries.

The main kernel `breed_kernel` uses the library `nubma.cuda`. With the help of this library, the kernel is being compiled into the usual CUDA kernel, which then is being appropriately called in the `fit()` method. The kernel implements the crossing-over and mutation procedures, producing the new generation. Each thread computes a few members of the new generation, with all the threads covering the whole new generation.

The fitness calculation and the parents picking is done using `torch`. The trick is to treat Numba device arrays as Torch tensors (both representations point to the same location in memory, so no copying happens). This enables us to use high-level torch tensor operations (which utilize CUDA under the hood) to do the calculation in a short and beautiful way.

CPU version mimics the interface and the behavior of the GPU version, except no CUDA kernel is being compiled (a regular function call is used, the function is non-parallelized (`breed_cpu`)) and the Torch operations with tensors are substituted with the corresponding Numpy operations with arrays.

## System and Software Specifications 
(Google Colab was used)
* __CPU__: Intel(R) Xeon(R) CPU @ 2.30GHz 
* __GPU__: NVIDIA Tesla K80

----

* __Python__: 3.7.12
* __CUDA__: 11.2
* __Numba__: 0.51.2
* __Torch__: 1.10.0+cu111

## Launch
The classes implementing both GPU and CPU versions of the genetic algorithm as well as the corresponding launch and "post-production" sequences are shown in `GeneticAlgo.ipynb`.

## Results

|    |   Points count |   Genes (x-powers) count |   Individuals count |   Epochs count |   CPU time |   GPU time |   Accel (CPUt/GPUt) |   CPU min err |   GPU min err |
|---:|---------------:|-------------------------:|--------------------:|---------------:|-----------:|-----------:|--------------------:|--------------:|--------------:|
|  0 |            100 |                        5 |                1000 |            100 |    3.07653 |   0.589349 |             5.22021 |     477.589   |     20.0089   |
|  1 |            500 |                        5 |                1000 |            100 |    3.30497 |   0.169656 |            19.4804  |      10.8938  |      9.41366  |
|  2 |           1000 |                        5 |                1000 |            100 |    3.4996  |   0.174744 |            20.027   |    2006       |      9.29086  |
|  3 |            100 |                        5 |                1000 |            500 |   15.5097  |   0.686715 |            22.5854  |     497.568   |     24.9102   |
|  4 |            500 |                        5 |                1000 |            500 |   16.4761  |   0.7096   |            23.2189  |      62.7141  |      6.13137  |
|  5 |           1000 |                        5 |                1000 |            500 |   17.752   |   0.709042 |            25.0366  |       5.22947 |      8.92565  |
|  6 |            100 |                        5 |                1000 |           1000 |   31.1414  |   1.34481  |            23.1568  |       5.56447 |   2253.82     |
|  7 |            500 |                        5 |                1000 |           1000 |   32.8537  |   1.40169  |            23.4386  |    3831.21    |   5199.62     |
|  8 |           1000 |                        5 |                1000 |           1000 |   35.8866  |   1.35805  |            26.425   |      11.8236  |      5.96198  |
|  9 |            100 |                        5 |                1500 |            100 |    4.70839 |   0.160673 |            29.3042  |       7.84104 |      0.815907 |
| 10 |            500 |                        5 |                1500 |            100 |    4.98048 |   0.165347 |            30.1214  |     220.309   |      4.70221  |
| 11 |           1000 |                        5 |                1500 |            100 |    5.35057 |   0.16847  |            31.7598  |      19.4271  |      7.80774  |
| 12 |            100 |                        5 |                1500 |            500 |   23.2704  |   0.671609 |            34.6487  |       4.33926 |      4.07093  |
| 13 |            500 |                        5 |                1500 |            500 |   25.0128  |   0.707292 |            35.3641  |      67.9697  |      4.16318  |
| 14 |           1000 |                        5 |                1500 |            500 |   27.344   |   0.715023 |            38.2422  |       4.10096 |      5.15177  |
| 15 |            100 |                        5 |                1500 |           1000 |   46.2442  |   1.35132  |            34.2215  |       2.97195 |      3.35306  |
| 16 |            500 |                        5 |                1500 |           1000 |   49.6735  |   1.35136  |            36.7581  |    9949.27    |   7578.33     |
| 17 |           1000 |                        5 |                1500 |           1000 |   53.5368  |   1.35121  |            39.6214  |     483.027   |     42.1604   |
| 18 |            100 |                        5 |                2000 |            100 |    6.12596 |   0.168029 |            36.4577  |       3.2248  |      1.17798  |
| 19 |            500 |                        5 |                2000 |            100 |    6.71772 |   0.166865 |            40.2584  |       6.33817 |      8.17887  |
| 20 |           1000 |                        5 |                2000 |            100 |    7.45404 |   0.175122 |            42.5648  |      10.2182  |     24.2422   |
| 21 |            100 |                        5 |                2000 |            500 |   31.2937  |   0.676172 |            46.2807  |       1.91062 |      3.24824  |
| 22 |            500 |                        5 |                2000 |            500 |   33.826   |   0.712601 |            47.4684  |     813.206   |      4.87118  |
| 23 |           1000 |                        5 |                2000 |            500 |   37.3463  |   0.733053 |            50.9462  |    2412.1     |    969.458    |
| 24 |            100 |                        5 |                2000 |           1000 |   62.4991  |   1.34984  |            46.3009  |       2.49218 |      2.05579  |
| 25 |            500 |                        5 |                2000 |           1000 |   67.3864  |   1.36956  |            49.203   |    1226.7     |      5.93685  |
| 26 |           1000 |                        5 |                2000 |           1000 |   74.5575  |   1.37915  |            54.0605  |       6.06891 |      4.14703  |


### *Time*
If we were to watch one of the *Points count*, *Genes count*, *Individuals count*, *Epochs count* parameters and lock the three others, we would see that the CPU time is growing with that parameter growing. For the GPU time that is basically the same except for few small cases.

### *Acceleration*
We can see that the acceleration is significant.

If we were to watch one of the *Points count*, *Genes count*, *Individuals count*, *Epochs count* parameters and lock the three others, we would see that the acceleration is growing with that parameter growing.

### *Best GPU result*
The visualisation of the approximation obtained on the GPU in the #9 experiment is shown in the next figure. That is the experiment where the smallest error value was obtained.

![image](https://user-images.githubusercontent.com/43893150/148715451-7ba98fb5-94fe-48ac-8ba7-23045d0b1d67.png)

## Conclusion
The experiment allowed us to see the practical benefits of using GPU in the implementation of a genetic algorithm. Taking into account the fact that the genetic algorithm is "random" in its basis and can converge for a long time, we can conclude that it is worth implementing it on the GPU in terms of time.
