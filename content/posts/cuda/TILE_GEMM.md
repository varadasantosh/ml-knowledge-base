
---
title: "Tiled GEMM (Small Matrices)"
---

<!--more-->

# Introduction to GEMM With Tiling

Matrix multiplication is a foundational operation in scientific computing and deep learning. In the GPU context, optimizing matrix multiplication can lead to significant performance gains. A naive implementation, however, has multiple limitations related to memory access latency and inefficient data reuse.


# Limitations of Naive Matrix Multiplication

- Each element required during multiplication is fetched from **global memory**, the slowest memory in the GPU hierarchy.
- Memory hierarchy: **Registers > Shared Memory > Global Memory** (in decreasing order of speed and increasing order of proximity to threads).
- There are many instances where the same element is needed by multiple threads. For example, to compute both C(0,0) and C(0,1), we need A(0,0)
- Without shared memory, every thread accesses global memory even if the data is common. Shared memory solves this by allowing data reuse among threads within the same thread block, reducing global memory transactions.

       ## Example:   
        C(0,0) = A(0,0) * B(0,0) + A(0,1) * B(1,0) + A(0,2) * B(2,0) + A(0,3) * B(3,0) 
        C(0,1) = A(0,0) * B(0,1) + A(0,1) * B(1,1) + A(0,2) * B(2,1) + A(0,3) * B(3,1)
   
# Why Tiling:    
Shared Memory size is limites for instance A matrix of size `512*512` with Full precision (FP32) which occupies 4 Bytes, requires ``` (512*512*4)/1024 = 1024 KB ``` , for Ampere Architecture the maximum size of shared memory is `192 KB` per SM. which is much less than the shared Memory. 
Hence we can't fit int the whole matrix into shared memory.
**Solution**:- Split matrices into smaller submatrices (tiles) and load them in phases into shared memory to perform computations more efficiently.

# Matrix Multiplication with Tiling

Let us consider Matrix of size `4*4` &  `TILE_SIZE` 2 , this divides the matrices into 4 submatrices(tiles) of each `2*2` size like show in below, each tile in output matrix is computed by one Thread Block. Each block further contains 4 Threads. 

**Note** while launching kernel we need to ensure that we allocate number of thread blocks equivalent to number of Tiles,also each block should have 4 threads as each tiles comprises of 2*2 matrices that results in 4 numbers in each tile, to work on each element of the tile we need one thread hence we would need 4 threads, in this example we are considering (2,2) block that has 2 threads in y 
dimension and 2 threads in x dimension 

## Block Layout 
     
     blockIdx.x=[0,1] , blockIdx.y=[0,1]

## Thread Layout     

    threadIdx.y=[0,1] threadIdx.x=[0,1]


# Splitting Matrices into Tiles

![overview.jpeg](/images/cuda/mma/tiling/6ede56f3-117e-4fd0-b6a7-2e1efff1dcb4.jpeg)


- Tile C(0,0) = A(0,0) * B(0,0) + A(0,1) * B(1,0) => Thread Block (0,0)
- Tile C(0,1) = A(0,0) * B(0,1) + A(0,1) * B(1,1) => Thread Block (0,1)
- Tile C(1,0) = A(1,0) * B(0,0) + A(1,1) * B(1,0) => Thread Block (1,0)
- Tile C(1,1) = A(1,0) * B(0,1) + A(1,1) * B(1,1) => Thread Block (1,1)


## Tile C(0,0)

![image.png](/images/cuda/mma/tiling/image.png)

## Tile C(0,1)

![image-2.png](/images/cuda/mma/tiling/image-2.png)

## Tile C(1,0)

![image-3.png](/images/cuda/mma/tiling/image-3.png)

## Tile C(1,1)

![image-4.png](/images/cuda/mma/tiling/image-4.png)


The process of Loading Tiles is further divided into several phases, if we observe the Tile C(0,0) it needs **2** tiles from A & **2** tiles from B, hence we divide loading tiles into **2** phases , we load one tile from each A & B in phase-1, and next set of tiles in phase-2, this happens parallely across 4 thread Blocks, during phase-1 the partial results of the corresponding block are calculated, at the end of phase -2 once all blocks are done with computation , we arriva at final **C** matrix.

Below is the representation of what happens during phase-1 from each Thread Block perspective, just a note that we have two phases as we are dividing the target into `TILE_SIZE=2`, which needs two tiles from both `A & B` 

Inorder for us to Load the tiles in different phases , we need a formula to calculate which elements or indices needs to be fetched from Global Memory to Shared Memory for the respective tiles of A & B. Below are the formula and visual representation of tiles being loaded and calculated
in each phase

```
   blockDim.x= 2, blockDimy=2 
   row = blockIdx.y*blockDim.y+threadIdx.y
   col = blockIdx.x*blockDim.x+threadIdx.x

   TILE_A = row*N +(t*TIL_SIZE+threadIdx.x)
   TILE_B = (t*TILE_SIZE+threadIdx.y)*N + col

   t = [0,1] => Represents Phases => Equivalent to Number of Tiles from A & B required to calculate Tile C 
   TILE_SIZE = 2

```

# Visual Represenation of Tile Loading

   ### Phase-1

   ![image-5.png](/images/cuda/mma/tiling/image-5.png)

   ![image-6.png](/images/cuda/mma/tiling/image-6.png)

   ![image-7.png](/images/cuda/mma/tiling/image-7.png)

   ![image-8.png](/images/cuda/mma/tiling/image-8.png)

   ### Phase-2

   ![image-9.png](/images/cuda/mma/tiling/image-9.png)

   ![image-10.png](/images/cuda/mma/tiling/image-10.png)

   ![image-11.png](/images/cuda/mma/tiling/image-11.png)

   ![image-12.png](/images/cuda/mma/tiling/image-12.png)
   


# Tile Index Computation

![image-15.png](/images/cuda/mma/tiling/image-15.png) 

## Indices of Matrices A & B in Row Major Layout

 **A**

 ![image-13.png](/images/cuda/mma/tiling/image-13.png)

 **B**

 ![image-16.png](/images/cuda/mma/tiling/image-16.png)

## Tile Calculation - Thread Block (0,0):

Where `t` ranges over phases: `t = [0, 1]`

### Example: Block (0,0), Phase-1 (t = 0)

| threadIdx | row | col | TILE\_A Index | TILE\_B Index |
| --------- | --- | --- | ------------- | ------------- |
| (0,0)     | 0   | 0   | 0             | 0             |
| (0,1)     | 0   | 1   | 1             | 1             |
| (1,0)     | 1   | 0   | 4             | 4             |
| (1,1)     | 1   | 1   | 5             | 5             |

### Phase-2 (t = 1)

| threadIdx | row | col | TILE\_A Index | TILE\_B Index |
| --------- | --- | --- | ------------- | ------------- |
| (0,0)     | 0   | 0   | 2             | 8             |
| (0,1)     | 0   | 1   | 3             | 9             |
| (1,0)     | 1   | 0   | 6             | 12            |
| (1,1)     | 1   | 1   | 7             | 13            |

## Step-by-Step Computation

   ### Phase-1 

   ```
      t=0

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (0,0)
      
         row = 0 * 2 + 0
         col = 0 * 2 + 0

      TILE_A_IDX = 0*4 + (0*2+0) = 0   
      TILE_B_IDX = (0*2+0) * 4 + 0 = 0

   Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (0,1)
      
         row = 0 * 2 + 0 = 0
         col = 0 * 2 + 1 = 1

      TILE_A_IDX = 0*4 + (0*2+1) = 1   
      TILE_B_IDX = (0*2+0) * 4 + 1 = 1

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (1,0)
      
         row = 0 * 2 + 1 = 1
         col = 0 * 2 + 0 = 0 

      TILE_A_IDX = 1*4 + (0*2+0) = 4   
      TILE_B_IDX = (0*2+1) * 4 + 0 = 4  

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (1,1)
      
         row = 0 * 2 + 1 = 1
         col = 0 * 2 + 1 = 1

      TILE_A_IDX = 1*4 + (0*2+1) = 5   
      TILE_B_IDX = (0*2+1) * 4 + 1 = 5   
   

   ```

   **TILE_A INDICES - (0,1,4,5)**
   **TILE_B INDICES - (0,1,4,5)**
 
   
   ![image-17.png](/images/cuda/mma/tiling/image-17.png) 

### Phase-2

   ```

      t = 1

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (0,0)
      
         row = 0 * 2 + 0
         col = 0 * 2 + 0

      TILE_A_IDX = 0*4 + (1*2+0) = 2
      TILE_B_IDX = (1*2+0) * 4 + 0 = 8

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (0,1)
      
         row = 0 * 2 + 0 = 0
         col = 0 * 2 + 1 = 1

      TILE_A_IDX = 0*4 + (1*2+1) = 3  
      TILE_B_IDX = (1*2+0) * 4 + 1 = 9

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (1,0)
      
         row = 0 * 2 + 1 = 1
         col = 0 * 2 + 0 = 0 

      TILE_A_IDX = 1*4 + (1*2+0) = 6   
      TILE_B_IDX = (1*2+1) * 4 + 0 = 12 

      Thread Block =>(blockIdx.y,blockIdx.x)=(0,0) & Threads => (threadIdx.y,threadIdx.x) = (1,1)
      
         row = 0 * 2 + 1 = 1
         col = 0 * 2 + 1 = 1

      TILE_A_IDX = 1*4 + (1*2+1) = 7  
      TILE_B_IDX = (1*2+1) * 4 + 1 = 13 

```
 **TILE_A INDICES - (2,3,6,7)**
 **TILE_B INDICES - (8,9,12,13)**

   
  ![image-19.png](/images/cuda/mma/tiling/image-18.png)

The above described is an example of Tile Calculation for Block(0,0), tiles are loaded to other blocks in similar fashion in two phases and each block calculates the final tile it is designated to thus we final arrive at Matrix Multiplication by using shared memory