# Lab #1 - Vector Sum
## Rail Gabbasov, 6132-010402D

## Description
In this work, a comparison of the following vector sum finding methods is done:
- Cython CUDA
- Cython
- NumPy

CUDA version uses tree-based approach with sequential adressing in shared memory. First addition as performed during loading of the vector into shared memory.

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

# Results

## Lenghts are powers of 10

### *Time (ms)*
|    N |       CPU |     NumPy |       GPU |
|:-----|----------:|----------:|----------:|
| 10^3 | 0.0052006 | 0.02084   | 0.0176832 |
| 10^4 | 0.013056  | 0.0241806 | 0.0159616 |
| 10^5 | 0.104746  | 0.160905  | 0.0153088 |
| 10^6 | 0.480154  | 0.701885  | 0.0201408 |
| 10^7 | 5.55098   | 8.26024   | 0.0999808 |

### *Time graphs*
todo

### *Acceleration graphs*
todo

## Lenghts are powers of 2

### *Time (ms)*
|    N |         CPU |      NumPy |       GPU |
|:-----|------------:|-----------:|----------:|
| 2^10 |  0.00244821 |  0.0136666 | 0.0108032 |
| 2^11 |  0.00295104 |  0.048239  | 0.0109696 |
| 2^12 |  0.00479883 |  0.010756  | 0.0112128 |
| 2^13 |  0.00786739 |  0.0150646 | 0.008096  |
| 2^14 |  0.00985158 |  0.0177864 | 0.0075072 |
| 2^15 |  0.0184298  |  0.0299562 | 0.010432  |
| 2^16 |  0.0354864  |  0.05376   | 0.0133312 |
| 2^17 |  0.0633006  |  0.0954564 | 0.0143552 |
| 2^18 |  0.123436   |  0.194937  | 0.0140928 |
| 2^19 |  0.250982   |  0.379408  | 0.02064   |
| 2^20 |  0.490977   |  0.739468  | 0.0201664 |
| 2^21 |  1.00356    |  1.46698   | 0.0328256 |
| 2^22 |  2.30615    |  3.31979   | 0.0500032 |
| 2^23 |  4.58195    |  6.86304   | 0.0862912 |
| 2^24 |  9.35645    | 13.9229    | 0.158752  |
| 2^25 | 18.411      | 28.326     | 0.303661  |

### *Time graphs*
todo

### *Acceleration graphs*
todo

## Conclusion
We can note that NumPy version is the slowest of all everywhere. A probable reason for this is that this is an implementation in Python, and the other two implementations are basically C-based.
The acceleration of the GPU version in comparison to CPU is growing with N growing.
The GPU version starts to dominate by N = 2^13 ≈ 10^4.
GPU time starts to rise by N = 2^20 ≈ 10^6.
