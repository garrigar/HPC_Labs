# Lab #0 - Matrix Multiplication
## Rail Gabbasov, 6132-010402D

## Description
In this work, a comparison of the following matrix multiplication methods is done:
- Cython CUDA (cuBLAS)
- Cython
- Cython OpenMP

## System and Software Specifications 
* __OS__:  Ubuntu 20.04.3 LTS 
* __Kernel__:  4.15.0-163-generic 
* __CPU__: Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz 
* __Memory__: 128 GB
* __GPUs__: NVIDIA GeForce RTX 2080 Ti

----

* __Python__: 3.8.10
* __CUDA__: 11.3
* __cuBLAS__: 11.5.1

## Launch
The whole launch sequence is shown in `main.ipynb`.

## Results

### *Time (ms)*
|    N |            CPU |   CPU OPENMP |       GPU |
|-----:|---------------:|-------------:|----------:|
|    8 |     0.00474341 |     0.163313 | 0.0141568 |
|   16 |     0.010795   |     0.122105 | 0.0131904 |
|   32 |     0.0580452  |     0.128014 | 0.0161856 |
|   64 |     0.317804   |     0.175761 | 0.0141504 |
|  128 |     5.03566    |     0.576763 | 0.02528   |
|  256 |    22.0501     |     5.75079  | 0.0251392 |
|  512 |   216.3        |    27.4725   | 0.0502912 |
| 1024 |  2635.73       |   305.528    | 0.232941  |
| 2048 | 27985.7        |  2744.52     | 1.62712   |

### *Time graphs*
![image](https://user-images.githubusercontent.com/43893150/148308482-9be7601c-ebe6-4ae2-820c-989b85454736.png)

### *Acceleration graphs*
![image](https://user-images.githubusercontent.com/43893150/148308528-9543269c-d795-4713-b2ef-5fd441cf8a86.png)

## Conclusion
GPU version is obviously the fastest of all versions. OpenMP tackles the problem pretty well up until N=256, but is slower afterwards.
