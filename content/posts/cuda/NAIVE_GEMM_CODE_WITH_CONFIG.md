
---
title: "Naive Matrix Multiplication"
---

<!--more-->

# Install NVCC For Jupyter


```python
pip install nvcc4jupyter
```

    Requirement already satisfied: nvcc4jupyter in /system/conda/miniconda3/envs/cloudspace/lib/python3.10/site-packages (1.2.1)
    Note: you may need to restart the kernel to use updated packages.


# Load the NVCC Extension


```python
%load_ext nvcc4jupyter
```

    Source files will be saved in "/tmp/tmp9pqtvywp".


# Matrix Multipliation with 1D Grid & 1D Block


```python
%%cuda

#include <stdio.h>

#define N 8

__global__ void gemm(int* dA, int* dB, int* dC, int mat_dim)
{

        int threadId = (blockIdx.x*blockDim.x) +threadIdx.x;

        int sum= 0;
        int row  = threadId/mat_dim;
        int col  = threadId%mat_dim; 

        if (row < N && col < N)
        {
            for (int k=0 ; k<mat_dim; k++)
            {
              sum = sum + (dA[(row*mat_dim)+k] * dB[(k*mat_dim)+col]);
             }
           dC[threadId] = sum;
        }

}

__host__ int main(){

   printf("GEMM Started\n");     
   int* A;
   int* B;
   int* C;

   int* dA;
   int* dB;
   int* dC;

   A = (int*) malloc(N*N*sizeof(int));
   B = (int*) malloc(N*N*sizeof(int));
   C = (int*) malloc(N*N*sizeof(int));

  cudaMalloc((void**)&dA, N*N * sizeof(int));
  cudaMalloc((void**)&dB, N*N * sizeof(int));
  cudaMalloc((void**)&dC, N*N * sizeof(int));

   for (int i=0; i<N; i++) 
   {
       for (int j=0; j<N; j++) 
       {
           A[i*N+j] = i*N+j;
           B[i*N+j] = i*N+j;
        }
   } 

   cudaMemcpy(dA, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(dB, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

   
   gemm<<<1,N*N>>>(dA,dB,dC,N);

   cudaDeviceSynchronize();

   cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    } 
   printf("GEMM Completed Successfully\n");

   cudaMemcpy(C,dC, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++)
  {
      for (int j=0; j<N; j++)
    {
        printf("%u    ",C[i*N+j]);
    }
     printf("\n");
   }

   cudaFree(dA);
   cudaFree(dB);
   cudaFree(dC);
   
   free(A);
   free(B);
   free(C);

}
```

    GEMM Started
    GEMM Completed Successfully
    1120    1148    1176    1204    1232    1260    1288    1316    
    2912    3004    3096    3188    3280    3372    3464    3556    
    4704    4860    5016    5172    5328    5484    5640    5796    
    6496    6716    6936    7156    7376    7596    7816    8036    
    8288    8572    8856    9140    9424    9708    9992    10276    
    10080    10428    10776    11124    11472    11820    12168    12516    
    11872    12284    12696    13108    13520    13932    14344    14756    
    13664    14140    14616    15092    15568    16044    16520    16996    
    


# Matrix Multiplications using using 2D Grid & 2D Block


```python
%%cuda

#include <stdio.h>
#define N 8

__global__ void gemm(int* dA, int* dB, int* dC, int matDim){

    
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int sum =0;

    if(row<N && col<N)
    {
        for (int k=0;k<N;k++){
    
            sum = sum + dA[row*N+k] * dB[k*N+col]; 
        
        }  
    
         dC[row*N+col] = sum;
    }

}


__host__ int main(){


    int *A, *B, *C;
    int *dA, *dB, *dC;

    A = (int*) malloc(N*N*sizeof(int));
    B = (int*) malloc(N*N*sizeof(int));
    C = (int*) malloc(N*N*sizeof(int));

    cudaMalloc( (void**)&dA, N*N*sizeof(int));
    cudaMalloc( (void**)&dB, N*N*sizeof(int));
    cudaMalloc( (void**)&dC, N*N*sizeof(int)); 

    for (int i=0; i<N; i++)
   {
     for (int j=0; j<N; j++ )
     {
        A[i*N+j] = i*N+j;
        B[i*N+j] = i*N+j; 
          
      }  
    }

    cudaMemcpy(dA, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    int block_size= 4;
    dim3 blockDim(block_size,block_size);
    dim3 gridDim((N+block_size-1)/block_size, (N+block_size-1)/block_size);
    

    gemm<<<gridDim,blockDim>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();
    cudaMemcpy(C,dC,N*N*sizeof(int), cudaMemcpyDeviceToHost);


    for(int i=0; i<N; i++)
    {
      for(int j=0; j<N; j++)
      {
         printf("%u ", C[i*N+j]);
      }
       printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);
    
    

}
```

    1120 1148 1176 1204 1232 1260 1288 1316 
    2912 3004 3096 3188 3280 3372 3464 3556 
    4704 4860 5016 5172 5328 5484 5640 5796 
    6496 6716 6936 7156 7376 7596 7816 8036 
    8288 8572 8856 9140 9424 9708 9992 10276 
    10080 10428 10776 11124 11472 11820 12168 12516 
    11872 12284 12696 13108 13520 13932 14344 14756 
    13664 14140 14616 15092 15568 16044 16520 16996 
    


# Matrix Multiplications using using 1D Grid & 3D Block


```python
%%cuda

#include <stdio.h>
#define N 8
#define BLOCK_SIZE 4

__global__ void gemm(int* dA, int* dB, int* dC, int matDim){

    
    int localThreadId =  threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x+threadIdx.x;
    int globalThreadId = (blockIdx.x * blockDim.x* blockDim.y * blockDim.z) + localThreadId; 

    int row  = globalThreadId/matDim;
    int col  = globalThreadId%matDim;

    int sum =0;

    if(row<N && col<N)
   {
        for (int k=0;k<N;k++){
    
            sum = sum + dA[row*N+k] * dB[k*N+col]; 
        
        }  
    
         dC[row*N+col] = sum;
    } 
}


__host__ int main(){


    int *A, *B, *C;
    int *dA, *dB, *dC;

    A = (int*) malloc(N*N*sizeof(int));
    B = (int*) malloc(N*N*sizeof(int));
    C = (int*) malloc(N*N*sizeof(int));

    cudaMalloc( (void**)&dA, N*N*sizeof(int));
    cudaMalloc( (void**)&dB, N*N*sizeof(int));
    cudaMalloc( (void**)&dC, N*N*sizeof(int)); 

    for (int i=0; i<N; i++)
   {
     for (int j=0; j<N; j++ )
     {
        A[i*N+j] = i*N+j;
        B[i*N+j] = i*N+j; 
          
      }  
    }

    cudaMemcpy(dA, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE);
    dim3 gridDim((N+BLOCK_SIZE-1)/BLOCK_SIZE);
    

    gemm<<<gridDim,blockDim>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();
    cudaMemcpy(C,dC,N*N*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++)
    {
      for(int j=0; j<N; j++)
      {
         printf("%u ", C[i*N+j]);
      }
       printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(A);
    free(B);
    free(C);

}
```

    1120 1148 1176 1204 1232 1260 1288 1316 
    2912 3004 3096 3188 3280 3372 3464 3556 
    4704 4860 5016 5172 5328 5484 5640 5796 
    6496 6716 6936 7156 7376 7596 7816 8036 
    8288 8572 8856 9140 9424 9708 9992 10276 
    10080 10428 10776 11124 11472 11820 12168 12516 
    11872 12284 12696 13108 13520 13932 14344 14756 
    13664 14140 14616 15092 15568 16044 16520 16996 
    



```python

```
