

---
title: "CUDA Programming Flow"
---

<!--more-->

## ✅ Quick Summary: CUDA Programming Steps

        | Step | Action                   | Code/Function Used                                   | Notes                         |
        |------|--------------------------|------------------------------------------------------|-------------------------------|
        | 1    | Declare Host Variables   | `int *h_A = (int *)malloc(size);`                    | Use standard C/C++ memory allocation |
        | 2    | Declare Device Variables | `int *d_A;`                                           | Only declare pointers here   |
        | 3    | Allocate Device Memory   | `cudaMalloc((void**)&d_A, size);`                     | Allocates memory on GPU      |
        | 4    | Copy Host → Device       | `cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);` | Copy input data to GPU       |
        | 5    | Launch Kernel            | `myKernel<<<gridDim, blockDim>>>(...);`               | Set up grid/block dimensions |
        |      |                          | `cudaDeviceSynchronize();`                            | Ensure kernel execution completes |
        | 6    | Copy Device → Host       | `cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);` | Retrieve output from GPU      |
        | 7    | Free Memory              | `cudaFree(d_A); free(h_A);`                           | Cleanup to avoid memory leaks |
