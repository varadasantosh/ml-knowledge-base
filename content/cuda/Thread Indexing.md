


---
title: "GPU Thread Hierarchy and Indexing"
---
<!--more-->

GPUs execute many parallel threads. These threads are organized into a three-level hierarchy:

- **Grids:** The entire collection of blocks that execute a kernel.
- **Blocks:** Subdivisions of the grid that contain groups of threads. Threads within a block can cooperate via shared memory.
- **Threads:** The smallest unit of execution.

When launching a kernel (the function that performs computation), you specify this thread hierarchy using the following syntax:

```cpp
kernel_name<<<gridDim, blockDim>>>(...);

```

Here, gridDim and blockDim can be configured with 1, 2, or 3 dimensions. Their possible combinations include:

  - gridDim = 1D & blockDim = 1D
  - gridDim = 1D & blockDim = 2D
  - gridDim = 1D & blockDim = 3D   
  - gridDim = 2D & blockDim = 1D
  - gridDim = 2D & blockDim = 2D
  - gridDim = 2D & blockDim = 3D 
  - gridDim = 3D & blockDim = 1D
  - gridDim = 3D & blockDim = 2D
  - griddIM = 3D & blockDim = 3D 

# Thread Indexing – 1D Grid and 1D Block

Example: Grid with 2 Blocks, Each with 5 Threads
Grid configuration: (2 blocks, each with 5 threads), can be denoted as (2,5).

Local Thread Indexing:
Within each block, threads are uniquely indexed from 0 to 4.

## Local Thread Index

    | BlockID   | ThreadID (Local)|
    |---------  |-----------------|
    |  Block-0  |      Thread-0   |
    |  Block-0  |      Thread-1   |
    |  Block-0  |      Thread-2   |
    |  Block-0  |      Thread-3   |
    |  Block-0  |      Thread-4   |
    |  Block-1  |      Thread-0   |
    |  Block-1  |      Thread-1   |
    |  Block-1  |      Thread-2   |
    |  Block-1  |      Thread-3   |
    |  Block-1  |      Thread-4   |   

## Global ThreadIndex
  
  Although thread indices are unique within a block, they are not unique across blocks. To compute a global thread ID, we use this one-dimensional indexing formula:

  we take help of predefined variables , in this context we only have one dimension hence these variables threadIdx.x, blockIdx.x, blockDim.x 

   ``` Global threadId = blockIdx.x * blockDim.x + threadIdx.x ```


    1. blockDim.x= 5
    2. blockIdx.x = [0,1]
    3. threadIdx.x = [0,1,2,3,4]
    
    |   BlockID  |  Calculation |   Global ThreadID   |
    |------------|--------------|---------------------|
    |   Block-0  | 0 * 5 + 0    |           0         |
    |   Block-0  | 0 * 5 + 1    |           1         |
    |   Block-0  | 0 * 5 + 2    |           2         |
    |   Block-0  | 0 * 5 + 3    |           3         |
    |   Block-0  | 0 * 5 + 4    |           4         |
    |   Block-0  | 1 * 5 + 0    |           5         |
    |   Block-0  | 1 * 5 + 1    |           6         |   
    |   Block-0  | 1 * 5 + 2    |           7         |
    |   Block-0  | 1 * 5 + 3    |           8         |
    |   Block-0  | 1 * 5 + 4    |           9         |

# Thread Indexing – 2D Grid and 1D Block

When the grid of blocks is two-dimensional, additional parameters (blockIdx.x and blockIdx.y) define the block's position in the grid. In this case, each block still uses 1D thread indexing.

Example: Grid with a 2D Arrangement (2 columns × 3 rows) of Blocks

## Grid Layout :
   ------------

  Imagine your grid is arranged as 2 blocks in the x-dimension and 3 blocks in the y-dimension. The grid can be represented as:

    [ 
      [Block (0,0), Block (0,1)],
      [Block (1,0), Block (1,1)],
      [Block (2,0), Block (2,1)] 
    ]

  Total Blocks =2(columns)×3(rows)=6.


## Global Block ID Calculation

A unique global block ID is computed as: ``` blockId = blockIdx.y * gridDim.x + blockIdx.x ```

For this example (with gridDim.x = 2):

    |   blockIdx.y  |  blockIdx.x      |      Calculation      |  Global BlockID  |
    |---------------|------------------|-----------------------|------------------|
    |   0           |     0            |         0 * 2 + 0     |       0          |
    |   0           |     1            |         0 * 2 + 1     |       1          |
    |   1           |     0            |         1 * 2 + 0     |       2          |
    |   1           |     1            |         1 * 2 + 1     |       3          |
    |   2           |     0            |         2 * 2 + 0     |       4          |
    |   2           |     1            |         2 * 2 + 1     |       5          |

## Local Thread Indexing:
  
  Each block contains 5 threads with local thread indices ranging from 0 to 4.

Matrix Representation of 2D Blocks in the Grid    
    
    |       | x = 0        | x = 1        |
    |-------|--------------|--------------|
    | y=0   | Block (0,0)  | Block (0,1)  |
    | y=1   | Block (1,0)  | Block (1,1)  |
    | y=2   | Block (2,0)  | Block (2,1)  |


    |       | x = 0        | x = 1        |
    |-------|--------------|--------------|
    | y=0   |       0      |    1         |
    | y=1   |       2      |    3         |
    | y=2   |       4      |    5         |



## Global Thread Index Calculation

Once the unique global block ID is computed, the global thread ID is calculated by:

threadId = blockId * blockDim.x + threadIdx.x;

- blockDim.x = 5
- threadIdx.x = [0, 1, 2, 3, 4]



        |   GlobalBlockID   |  Calculation     |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 5 + 0      |           0           |
        |       0           |   0 * 5 + 1      |           1           |
        |       0           |   0 * 5 + 2      |           2           |
        |       0           |   0 * 5 + 3      |           3           |
        |       0           |   0 * 5 + 4      |           4           |
        |       1           |   1 * 5 + 0      |           5           |
        |       1           |   1 * 5 + 1      |           6           |
        |       1           |   1 * 5 + 2      |           7           |
        |       1           |   1 * 5 + 3      |           8           |
        |       1           |   1 * 5 + 4      |           9           |
        |       2           |   2 * 5 + 0      |           10          |
        |       2           |   2 * 5 + 1      |           11          |
        |       2           |   2 * 5 + 2      |           12          |
        |       2           |   2 * 5 + 3      |           13          |
        |       2           |   2 * 5 + 4      |           14          |
        |       3           |   3 * 5 + 0      |           15          |
        |       3           |   3 * 5 + 1      |           16          |
        |       3           |   3 * 5 + 2      |           17          |
        |       3           |   3 * 5 + 3      |           18          |
        |       3           |   3 * 5 + 4      |           19          |
        |       4           |   4 * 5 + 0      |           20          |
        |       4           |   4 * 5 + 1      |           21          |
        |       4           |   4 * 5 + 2      |           22          |
        |       4           |   4 * 5 + 3      |           23          |
        |       4           |   4 * 5 + 4      |           24          |
        |       5           |   5 * 5 + 0      |           25          |
        |       5           |   5 * 5 + 1      |           26          |
        |       5           |   5 * 5 + 2      |           27          |
        |       5           |   5 * 5 + 3      |           28          |
        |       5           |   5 * 5 + 4      |           29          |


# Thread Indexing – 3D Grid and 1D Block

When the grid of blocks is three-dimensional, we have predefined parameters (blockIdx.x and blockIdx.y, blockIdx.z) define the block's position in the grid. In this case, each block still uses 1D thread indexing.

Example: Grid with a 3D Arrangement (2 × 3 * 4) of Blocks

(x,y,z) = (2,3,4)

## Grid Layout:
--------------


    Total Blocks =2(x)×3(y)*4(z) = 24.

    grid = [

    [  # z = 0
        [ Block(0,0,0), Block(0,0,1) ],  # y = 0
        [ Block(0,1,0), Block(0,1,1) ],  # y = 1
        [ Block(0,2,0), Block(0,2,1) ]   # y = 2
    ],
    [  # z = 1
        [ Block(1,0,0), Block(1,0,1) ],  # y = 0
        [ Block(1,1,0), Block(1,1,1) ],  # y = 1
        [ Block(1,2,0), Block(1,2,1) ]   # y = 2
    ],
    [  # z = 2
        [ Block(2,0,0), Block(2,0,1) ],  # y = 0
        [ Block(2,1,0), Block(2,1,1) ],  # y = 1
        [ Block(2,2,0), Block(2,2,1) ]   # y = 2
    ],
    [  # z = 3
        [ Block(3,0,0), Block(3,0,1) ],  # y = 0 
        [ Block(3,1,0), Block(3,1,1) ],  # y = 1
        [ Block(3,2,0), Block(3,2,1) ]   # y = 2
    ]
    ]

## Global Block ID Calculation

A unique global block ID is computed as: ``` blockId = blockIdx.z * gridDim.y * gridDim.x  + blockIdx.y * gridDim.x + blockIdx.x ```

For this example     
    
    gridDim.x = 2
    gridDim.y = 3
    gridDim.z = 4 


    |    blockIdx.z  |   blockIdx.y  |  blockIdx.x      |      Calculation           |  Global BlockID   |
    |----------------|---------------|------------------|--------------------------- | ------------------|
    |       0        |     0         |     0            |     0 * 2 * 3 + 0 * 2 + 0  |         0         |
    |       0        |     0         |     1            |     0 * 2 * 3 + 0 * 2 + 1  |         1         |
    |       0        |     1         |     0            |     0 * 2 * 3 + 1 * 2 + 0  |         2         |
    |       0        |     1         |     1            |     0 * 2 * 3 + 1 * 2 + 1  |         3         |
    |       0        |     2         |     0            |     0 * 2 * 3 + 2 * 2 + 0  |         4         |
    |       0        |     2         |     1            |     0 * 2 * 3 + 2 * 2 + 1  |         5         |
    |       1        |     0         |     0            |     1 * 2 * 3 + 0 * 2 + 0  |         6         |
    |       1        |     0         |     1            |     1 * 2 * 3 + 0 * 2 + 1  |         7         |
    |       1        |     1         |     0            |     1 * 2 * 3 + 1 * 2 + 0  |         8         |
    |       1        |     1         |     1            |     1 * 2 * 3 + 1 * 2 + 1  |         9         |
    |       1        |     2         |     0            |     1 * 2 * 3 + 2 * 2 + 0  |         10        |
    |       1        |     2         |     1            |     1 * 2 * 3 + 2 * 2 + 1  |         11        |
    |       2        |     0         |     0            |     2 * 2 * 3 + 0 * 2 + 0  |         12        |
    |       2        |     0         |     1            |     2 * 2 * 3 + 0 * 2 + 1  |         13        |
    |       2        |     1         |     0            |     2 * 2 * 3 + 1 * 2 + 0  |         14        |
    |       2        |     1         |     1            |     2 * 2 * 3 + 1 * 2 + 1  |         15        |
    |       2        |     2         |     0            |     2 * 2 * 3 + 2 * 2 + 0  |         16        |
    |       2        |     2         |     1            |     2 * 2 * 3 + 2 * 2 + 1  |         17        |
    |       3        |     0         |     0            |     3 * 2 * 3 + 0 * 2 + 0  |         18        |
    |       3        |     0         |     1            |     3 * 2 * 3 + 0 * 2 + 1  |         19        |
    |       3        |     1         |     0            |     3 * 2 * 3 + 1 * 2 + 0  |         20        |
    |       3        |     1         |     1            |     3 * 2 * 3 + 1 * 2 + 1  |         21        |
    |       3        |     2         |     0            |     3 * 2 * 3 + 2 * 2 + 0  |         22        |
    |       3        |     2         |     1            |     3 * 2 * 3 + 2 * 2 + 1  |         23        |



## Local Thread Indexing:
  
  Each block contains 5 threads with local thread indices from 0 to 4.


## Global Thread Index Calculation

Once the unique global block ID is computed, the global thread ID is calculated by:

threadId = blockId * blockDim.x + threadIdx.x;

- BlockDim.x = 5
- threadIdx.x = [0, 1, 2, 3, 4]

        |   GlobalBlockID   |  Calculation     |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 5 + 0      |           0           |
        |       0           |   0 * 5 + 1      |           1           |
        |       0           |   0 * 5 + 2      |           2           |
        |       0           |   0 * 5 + 3      |           3           |
        |       0           |   0 * 5 + 4      |           4           |
        |       1           |   1 * 5 + 0      |           5           |
        |       1           |   1 * 5 + 1      |           6           |
        |       1           |   1 * 5 + 2      |           7           |
        |       1           |   1 * 5 + 3      |           8           |
        |       1           |   1 * 5 + 4      |           9           |
        |       2           |   2 * 5 + 0      |           10          |
        |       2           |   2 * 5 + 1      |           11          |
        |       2           |   2 * 5 + 2      |           12          |
        |       2           |   2 * 5 + 3      |           13          |
        |       2           |   2 * 5 + 4      |           14          |
        |       3           |   3 * 5 + 0      |           15          |
        |       3           |   3 * 5 + 1      |           16          |
        |       3           |   3 * 5 + 2      |           17          |
        |       3           |   3 * 5 + 3      |           18          |
        |       3           |   3 * 5 + 4      |           19          |
        |       4           |   4 * 5 + 0      |           20          |
        |       4           |   4 * 5 + 1      |           21          |
        |       4           |   4 * 5 + 2      |           22          |
        |       4           |   4 * 5 + 3      |           23          |
        |       4           |   4 * 5 + 4      |           24          |
        |       5           |   5 * 5 + 0      |           25          |
        |       5           |   5 * 5 + 1      |           26          |
        |       5           |   5 * 5 + 2      |           27          |
        |       5           |   5 * 5 + 3      |           28          |
        |       5           |   5 * 5 + 4      |           29          |
        |       6           |   6 * 5 + 0      |           30          |
        |       6           |   6 * 5 + 1      |           31          |
        |       6           |   6 * 5 + 2      |           32          |
        |       6           |   6 * 5 + 3      |           33          |
        |       6           |   6 * 5 + 4      |           34          |
        |                   |                  |                       |
        |                   |                  |                       |
        |                   |                  |                       |
        |       23          |   23 * 5 + 0     |           115         |
        |       23          |   23 * 5 + 1     |           116         |
        |       23          |   23 * 5 + 2     |           117         |
        |       23          |   23 * 5 + 3     |           118         |
        |       23          |   23 * 5 + 4     |           119         |


# Thread Indexing – 1D Grid and 2D Block 
 
 - GridDim is  1D : gridDim.x = N (say gridDim.x = 1)
 - BlockDim is 2D : blockDim.x = 2, blockDim.y = 3

 Since GridDim is 1D, we have only one block. If gridDim.x = 1, the only block is blockIdx.x = 0.

    blockDim = (2, 3)
    ---------------------
    | (0,0) | (0,1) |
    | (1,0) | (1,1) |
    | (2,0) | (2,1) |
    ---------------------

To calculate global thread indices in 2D, we use:

int global_x = threadIdx.x + blockIdx.x * blockDim.x;
int global_y = threadIdx.y + blockIdx.y * blockDim.y;


Each thread in the single block Block(0) has a unique (threadIdx.x, threadIdx.y):

Global Thread Index (flattened row-wise):

    ---------------------
    |   0   |   1   |
    |   2   |   3   |
    |   4   |   5   |
    ---------------------


we can derive Global Index using Formulae ``` threadIdx.y * blockDim.x + threadIdx.x ```

    Block(0) – Threads with blockDim(2,3)

    blockDim.x = 2

    | threadIdx.y | threadIdx.x | GlobalThreadID   |
    |-------------|-------------|------------------|
    |     0       |     0       |    0 * 2 + 0     |
    |     0       |     1       |    0 * 2 + 1     |
    |     1       |     0       |    1 * 2 + 0     |
    |     1       |     1       |    1 * 2 + 1     |
    |     2       |     0       |    2 * 2  + 0    |
    |     2       |     1       |    2 * 2 + 1     |



# Thread Indexing - 2D Grid and 2D Block

dim3 gridDim(3,2) = (x,y)
dim3 blockDim(4,2) = (x,y)

gridDim.x = 3
gridDim.y = 2
blockDim.x = 4
blockDim.y = 2

Total Number of Blocks = 3*2 = 6 Blocks
Threads per Blocks = 4*2 = 8 Threads
Total Threads = 6 * 8 = 48 Threads

## Grid Layout
--------------

    gridDim = (3, 2)

    |        | x = 0 | x = 1 |  x=2      |
    |------- |-------|-------|-----------|
    |  y = 0 | (0,0) | (0,1) |  (0,2)    |
    |  y = 1 | (1,0) | (1,1) |  (1,2)    |
     ------------------------------------

    |       |  x = 0  |  x = 1  |  x = 2  |
    |-------|-------- |---------|---------|
    | y=0   |    0    |    1    |   2     |
    | y=1   |    3    |    4    |   5     |
    |-------------------------------------|


A unique global block ID is computed as: ``` blockId = blockIdx.y * gridDim.x + blockIdx.x ```

For this example (with gridDim.x = 2):

    |   blockIdx.y  |  blockIdx.x      |      Calculation      |  Global BlockId  |
    |---------------|------------------|-----------------------|------------------|
    |   0           |     0            |         0 * 3 + 0     |       0          |
    |   0           |     1            |         0 * 3 + 1     |       1          |
    |   0           |     2            |         0 * 3 + 2     |       2          |
    |   1           |     0            |         1 * 3 + 0     |       3          |
    |   1           |     1            |         1 * 3 + 1     |       4          |
    |   1           |     2            |         1 * 3 + 2     |       5          |


## Local Thread Indexing:
-----------------

    blockDim = (4, 2)

    |        | x = 0 | x = 1 |  x=2   | x=3  |
    |------- |-------|-------|--------|------|
    |  y = 0 | (0,0) | (0,1) |  (0,2) | (0,3)|
    |  y = 1 | (1,0) | (1,1) |  (1,2) | (1,3)|
     -----------------------------------------

    |       |  x = 0  |  x = 1  |  x = 2  | x = 3 |
    |-------|-------- |---------|---------|-------|
    | y=0   |    0    |    1    |   2     |   3   |
    | y=1   |    4    |    5    |   6     |   7   |
    |---------------------------------------------|

Local Thread Index - Linear Index can be calculated with formulae : ``` local threadId  = threadIdx.y * blockDim.x + threadIdx.x ```  

    |   threadIdx.y |  threadIdx.x     |      Calculation      |  Local ThreadId  |
    |---------------|------------------|-----------------------|------------------|
    |   0           |     0            |         0 * 4 + 0     |       0          |
    |   0           |     1            |         0 * 4 + 1     |       1          |
    |   0           |     2            |         0 * 4 + 2     |       2          |
    |   0           |     3            |         0 * 4 + 3     |       3          |
    |   1           |     0            |         1 * 4 + 0     |       4          |
    |   1           |     1            |         1 * 4 + 1     |       5          |
    |   1           |     2            |         1 * 4 + 2     |       6          |
    |   1           |     3            |         1 * 4 + 3     |       7          |


## Global Thread Index Calculation    

Finally using Global BlockId & Local ThreadId of each Block we can Calculate Global Thread Id using below formulae:

``` (blockId * blockDim.x * blockDim.y) + local threadId ```
            
``` blockId  = blockIdx.y * gridDim.x + blockIdx.x ```            
``` local ThreadId =  threadIdx.y * blockDim.x + threadIdx.x ```

  
        |   GlobalBlockID   |   LocalThreadId  |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 8 + 0      |           0           |
        |       0           |   0 * 8 + 1      |           1           |
        |       0           |   0 * 8 + 2      |           2           |
        |       0           |   0 * 8 + 3      |           3           |
        |       0           |   0 * 8 + 4      |           4           |
        |       0           |   0 * 8 + 5      |           5           |
        |       0           |   0 * 8 + 6      |           6           |
        |       0           |   0 * 8 + 7      |           7           |

        |       1           |   1 * 8 + 0      |           8           |
        |       1           |   1 * 8 + 1      |           9           |
        |       1           |   1 * 8 + 2      |           10          |
        |       1           |   1 * 8 + 3      |           11          |
        |       1           |   1 * 8 + 4      |           12          |
        |       1           |   1 * 8 + 5      |           13          |
        |       1           |   1 * 8 + 6      |           14          |
        |       1           |   1 * 8 + 7      |           15          |

        |       2           |   2 * 8 + 0      |           16          |
        |       2           |   2 * 8 + 1      |           17          |
        |       2           |   2 * 8 + 2      |           18          |
        |       2           |   2 * 8 + 3      |           19          |
        |       2           |   2 * 8 + 4      |           20          |
        |       2           |   2 * 8 + 5      |           21          |
        |       2           |   2 * 8 + 6      |           22          |
        |       2           |   2 * 8 + 7      |           23          |

      


        |       5           |   5 * 8 + 0      |           40          |
        |       5           |   5 * 8 + 1      |           41          |
        |       5           |   5 * 8 + 2      |           42          |
        |       5           |   5 * 8 + 3      |           43          |
        |       5           |   5 * 8 + 4      |           44          |
        |       5           |   5 * 8 + 5      |           45          |
        |       5           |   5 * 8 + 6      |           46          |
        |       5           |   5 * 8 + 7      |           47          |

        
# Thread Indexing - 3D Grid and 2D Block

dim3 gridDim(4,3,2) = (x,y,z)
dim3 blockDim(4,2) = (x,y) 

gridDim.x = 4
gridDim.y = 3
gridDim.z = 2

blockDim.x = 4
blockDim.y = 2

Number of Blocks = 24
Nubmer of Thread per Block = 8
Total Number of Threads  = 24 * 8 = 192

## Grid Layout
----------------
Grid Comprises of 2 matrices of (3*4) , which can be represented as below.

Total Number of Blocks = 4 * 3 * 2 = 24 Blocks

    grid = [

    [  # z = 0
        [ Block(0,0,0), Block(0,0,1), Block(0,0,2), Block(0,0,3) ],  # y = 0
        [ Block(0,1,0), Block(0,1,1), Block(0,1,2), Block(0,1,3) ],  # y = 1
        [ Block(0,2,0), Block(0,2,1), Block(0,2,2), Block(0,2,3) ]   # y = 2
    ],
    [  # z = 1
        [ Block(1,0,0), Block(1,0,1), Block(1,0,2), Block(1,0,3)  ],  # y = 0
        [ Block(1,1,0), Block(1,1,1), Block(1,1,2), Block(1,1,3)  ],  # y = 1
        [ Block(1,2,0), Block(1,2,1), Block(1,2,2), Block(1,2,3)  ]  # y = 2
    ]
    
    ]

A unique global block ID is computed as: ``` blockId = ( blockIdx.z * gridDim.y * gridDim.x ) + ( blockIdx.y * gridDim.x ) + blockIdx.x ```


        | blockIdx.z   |   blockIdx.y  |  blockIdx.x      |      Calculation                  |  Global BlockId  |
        |------------- |---------------|------------------|-----------------------------------|------------------|
        |      0       |   0           |     0            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 0  |       0          |
        |      0       |   0           |     1            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 1  |       1          |
        |      0       |   0           |     2            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 2  |       2          |
        |      0       |   0           |     3            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 3  |       3          |
        |      0       |   1           |     0            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 0  |       4          |
        |      0       |   1           |     1            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 1  |       5          |
        |      0       |   1           |     2            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 2  |       6          |
        |      0       |   1           |     3            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 3  |       7          |
        |      0       |   2           |     0            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 0  |       8          |
        |      0       |   2           |     1            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 1  |       9          |
        |      0       |   2           |     2            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 2  |       10         |
        |      0       |   2           |     3            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 3  |       11         |

        |      1       |   0           |     0            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 0  |       12          |
        |      1       |   0           |     1            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 1  |       13          |
        |      1       |   0           |     2            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 2  |       14          |
        |      1       |   0           |     3            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 3  |       15          |
        |      1       |   1           |     0            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 0  |       16          |
        |      1       |   1           |     1            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 1  |       17          |
        |      1       |   1           |     2            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 2  |       18          |
        |      1       |   1           |     3            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 3  |       19          |
        |      1       |   2           |     0            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 0  |       20          |
        |      1       |   2           |     1            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 1  |       21          |
        |      1       |   2           |     2            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 2  |       22          |
        |      1       |   2           |     3            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 3  |       23          |


## Local Thread Indexing:
-----------------

    blockDim = (4, 2)

    |        | x = 0 | x = 1 |  x=2   | x=3  |
    |------- |-------|-------|--------|------|
    |  y = 0 | (0,0) | (0,1) |  (0,2) | (0,3)|
    |  y = 1 | (1,0) | (1,1) |  (1,2) | (1,3)|
     -----------------------------------------

    |       |  x = 0  |  x = 1  |  x = 2  | x = 3 |
    |-------|-------- |---------|---------|-------|
    | y=0   |    0    |    1    |   2     |   3   |
    | y=1   |    4    |    5    |   6     |   7   |
    |---------------------------------------------|



Local Thread Index - Linear Index can be calculated with formulae : ``` local threadId  = threadIdx.y * blockDim.x + threadIdx.x ```  

    |   threadIdx.y |  threadIdx.x     |      Calculation      |  Local ThreadId  |
    |---------------|------------------|-----------------------|------------------|
    |   0           |     0            |         0 * 4 + 0     |       0          |
    |   0           |     1            |         0 * 4 + 1     |       1          |
    |   0           |     2            |         0 * 4 + 2     |       2          |
    |   0           |     3            |         0 * 4 + 3     |       3          |
    |   1           |     0            |         1 * 4 + 0     |       4          |
    |   1           |     1            |         1 * 4 + 1     |       5          |
    |   1           |     2            |         1 * 4 + 2     |       6          |
    |   1           |     3            |         1 * 4 + 3     |       7          |

## Global Thread Index Calculation:    

Finally using Global BlockId & Local ThreadId of each Block we can Calculate Global Thread Id using below formulae:

``` (blockId * blockDim.x * blockDim.y) + local threadId ```
            
``` blockId  = blockIdx.y * gridDim.x + blockIdx.x ```            
``` local ThreadId =  threadIdx.y * blockDim.x + threadIdx.x ```

  
        |   GlobalBlockID   |   LocalThreadId  |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 8 + 0      |           0           |
        |       0           |   0 * 8 + 1      |           1           |
        |       0           |   0 * 8 + 2      |           2           |
        |       0           |   0 * 8 + 3      |           3           |
        |       0           |   0 * 8 + 4      |           4           |
        |       0           |   0 * 8 + 5      |           5           |
        |       0           |   0 * 8 + 6      |           6           |
        |       0           |   0 * 8 + 7      |           7           |
        

        |       1           |   1 * 8 + 0      |           8           |
        |       1           |   1 * 8 + 1      |           9           |
        |       1           |   1 * 8 + 2      |           10          |
        |       1           |   1 * 8 + 3      |           11          |
        |       1           |   1 * 8 + 4      |           12          |
        |       1           |   1 * 8 + 5      |           13          |
        |       1           |   1 * 8 + 6      |           14          |
        |       1           |   1 * 8 + 7      |           15          |

        |       2           |   2 * 8 + 0      |           16          |
        |       2           |   2 * 8 + 1      |           17          |
        |       2           |   2 * 8 + 2      |           18          |
        |       2           |   2 * 8 + 3      |           19          |
        |       2           |   2 * 8 + 4      |           20          |
        |       2           |   2 * 8 + 5      |           21          |
        |       2           |   2 * 8 + 6      |           22          |
        |       2           |   2 * 8 + 7      |           23          |

      


        |       23          |   23 * 8 + 0      |           184         |
        |       23          |   23 * 8 + 1      |           185         |
        |       23          |   23 * 8 + 2      |           186         |
        |       23          |   23 * 8 + 3      |           187         |
        |       23          |   23 * 8 + 4      |           188         |
        |       23          |   23 * 8 + 5      |           189         |
        |       23          |   23* 8 + 6       |           190         |
        |       23          |   23 * 8 + 7      |           191         |


# Thread Indexing - 1D Grid and 3D Block

dim3 gridDim(1)
dim3 blockDim(6,3,2)  

(x,y,z) = (6,3,2)

gridDim.x = 1

blockDim.x = 6
blockDim.y = 3
blockDim.z = 2

Number of Threads per Block :- 6 * 3 * 2 = 36

## Grid Layout
--------------

Here , as we only have one Block we don't need to calculate GlobalID and directly we can start with calculating Local ThreadID inside
Block, which also becomes GlobalThreadID due to single Block

## Local Thread Indexing:
-----------------

   [

        [  # z = 0
            [ Thread(0,0,0), Thread(0,0,1), Thread(0,0,2), Thread(0,0,3), Thread(0,0,4), Thread(0,0,5)  ],  # y = 0
            [ Thread(0,1,0), Thread(0,1,1), Thread(0,1,2), Thread(0,1,3), Thread(0,1,4), Thread(0,1,5)  ],  # y = 1
            [ Thread(0,2,0), Thread(0,2,1), Thread(0,2,2), Thread(0,2,3), Thread(0,2,4), Thread(0,2,5)  ]   # y = 2
        ],
        [  # z = 1
            [ Thread(1,0,0), Thread(1,0,1), Thread(1,0,2), Thread(1,0,3), Thread(1,0,4), Thread(1,0,5)   ],  # y = 0
            [ Thread(1,1,0), Thread(1,1,1), Thread(1,1,2), Thread(1,1,3), Thread(1,1,4), Thread(1,1,5)   ],  # y = 1
            [ Thread(1,2,0), Thread(1,2,1), Thread(1,2,2), Thread(1,2,3), Thread(1,2,4), Thread(1,2,5)   ]  # y = 2
        ]
    
    ]

 Local ThreadID is computed as: ``` Local ThreadId = ( threadIdx.z * blockDim.y * blockDim.x ) + ( threadIdx.y * blockDim.x ) + threadIdx.x ```


        | threadIdx.z  |   threadIdx.y |  threadIdx.x     |      Calculation                  |  Local ThreadId  |
        |------------- |---------------|------------------|-----------------------------------|------------------|
        |      0       |   0           |     0            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 0  |       0          |
        |      0       |   0           |     1            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 1  |       1          |
        |      0       |   0           |     2            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 2  |       2          |
        |      0       |   0           |     3            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 3  |       3          |
        |      0       |   0           |     4            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 4  |       4          |
        |      0       |   0           |     5            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 5  |       5          |
        
        |      0       |   1           |     0            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 0  |       6          |
        |      0       |   1           |     1            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 1  |       7          |
        |      0       |   1           |     2            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 2  |       8          |
        |      0       |   1           |     3            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 3  |       9          |
        |      0       |   1           |     4            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 4  |       10         |
        |      0       |   1           |     5            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 5  |       11         |

        |      0       |   2           |     0            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 0  |       12         |
        |      0       |   2           |     1            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 1  |       13         |
        |      0       |   2           |     2            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 2  |       14         |
        |      0       |   2           |     3            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 3  |       15         |
        |      0       |   2           |     4            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 4  |       16         |
        |      0       |   2           |     5            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 5  |       17         |


        |      1       |   0           |     0            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 0  |       18         |
        |      1       |   0           |     1            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 1  |       19         |
        |      1       |   0           |     2            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 2  |       20         |
        |      1       |   0           |     3            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 3  |       21         |
        |      1       |   0           |     4            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 4  |       22         |
        |      1       |   0           |     5            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 5  |       23         |

        |      1       |   1           |     0            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 0  |       24         |
        |      1       |   1           |     1            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 1  |       25         |
        |      1       |   1           |     2            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 2  |       26         |
        |      1       |   1           |     3            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 3  |       27         |
        |      1       |   1           |     4            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 4  |       28         |
        |      1       |   1           |     5            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 5  |       29         |

        |      1       |   2           |     0            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 0  |       30         |
        |      1       |   2           |     1            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 1  |       31         |
        |      1       |   2           |     2            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 2  |       32         |
        |      1       |   2           |     3            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 3  |       33         |
        |      1       |   2           |     4            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 4  |       34         |
        |      1       |   2           |     5            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 5  |       35         |


# Thread Indexing - 2D Grid and 3D Block

dim3 gridDim(3,2) = (x,y)
dim3 blockDim(6,3,2)  = (x,y,z)

gridDim.x = 3
gridDim.y = 2

blockDim.x = 6
blockDim.y = 3
blockDim.z = 2

## Grid Layout
--------------

    gridDim = (3, 2)

    |        | x = 0 | x = 1 |  x=2      |
    |------- |-------|-------|-----------|
    |  y = 0 | (0,0) | (0,1) |  (0,2)    |
    |  y = 1 | (1,0) | (1,1) |  (1,2)    |
     ------------------------------------

    |       |  x = 0  |  x = 1  |  x = 2  |
    |-------|-------- |---------|---------|
    | y=0   |    0    |    1    |   2     |
    | y=1   |    3    |    4    |   5     |
    |-------------------------------------|


A unique global block ID is computed as: ``` blockId = blockIdx.y * gridDim.x + blockIdx.x ```

For this example (with gridDim.x = 2):

    |   blockIdx.y  |  blockIdx.x      |      Calculation      |  Global BlockId  |
    |---------------|------------------|-----------------------|------------------|
    |   0           |     0            |         0 * 3 + 0     |       0          |
    |   0           |     1            |         0 * 3 + 1     |       1          |
    |   0           |     2            |         0 * 3 + 2     |       2          |
    |   1           |     0            |         1 * 3 + 0     |       3          |
    |   1           |     1            |         1 * 3 + 1     |       4          |
    |   1           |     2            |         1 * 3 + 2     |       5          |

## Local Thread Indexing:
----------------------
 
 [

        [  # z = 0
            [ Thread(0,0,0), Thread(0,0,1), Thread(0,0,2), Thread(0,0,3), Thread(0,0,4), Thread(0,0,5)  ],  # y = 0
            [ Thread(0,1,0), Thread(0,1,1), Thread(0,1,2), Thread(0,1,3), Thread(0,1,4), Thread(0,1,5)  ],  # y = 1
            [ Thread(0,2,0), Thread(0,2,1), Thread(0,2,2), Thread(0,2,3), Thread(0,2,4), Thread(0,2,5)  ]   # y = 2
        ],
        [  # z = 1
            [ Thread(1,0,0), Thread(1,0,1), Thread(1,0,2), Thread(1,0,3), Thread(1,0,4), Thread(1,0,5)   ],  # y = 0
            [ Thread(1,1,0), Thread(1,1,1), Thread(1,1,2), Thread(1,1,3), Thread(1,1,4), Thread(1,1,5)   ],  # y = 1
            [ Thread(1,2,0), Thread(1,2,1), Thread(1,2,2), Thread(1,2,3), Thread(1,2,4), Thread(1,2,5)   ]  # y = 2
        ]
    
    ]

 Local ThreadID is computed as: ``` Local ThreadId = ( threadIdx.z * blockDim.y * blockDim.x ) + ( threadIdx.y * blockDim.x ) + threadIdx.x ```


        | threadIdx.z  |   threadIdx.y |  threadIdx.x     |      Calculation                  |  Local ThreadId  |
        |------------- |---------------|------------------|-----------------------------------|------------------|
        |      0       |   0           |     0            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 0  |       0          |
        |      0       |   0           |     1            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 1  |       1          |
        |      0       |   0           |     2            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 2  |       2          |
        |      0       |   0           |     3            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 3  |       3          |
        |      0       |   0           |     4            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 4  |       4          |
        |      0       |   0           |     5            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 5  |       5          |
        
        |      0       |   1           |     0            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 0  |       6          |
        |      0       |   1           |     1            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 1  |       7          |
        |      0       |   1           |     2            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 2  |       8          |
        |      0       |   1           |     3            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 3  |       9          |
        |      0       |   1           |     4            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 4  |       10         |
        |      0       |   1           |     5            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 5  |       11         |

        |      0       |   2           |     0            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 0  |       12         |
        |      0       |   2           |     1            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 1  |       13         |
        |      0       |   2           |     2            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 2  |       14         |
        |      0       |   2           |     3            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 3  |       15         |
        |      0       |   2           |     4            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 4  |       16         |
        |      0       |   2           |     5            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 5  |       17         |


        |      1       |   0           |     0            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 0  |       18         |
        |      1       |   0           |     1            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 1  |       19         |
        |      1       |   0           |     2            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 2  |       20         |
        |      1       |   0           |     3            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 3  |       21         |
        |      1       |   0           |     4            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 4  |       22         |
        |      1       |   0           |     5            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 5  |       23         |

        |      1       |   1           |     0            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 0  |       24         |
        |      1       |   1           |     1            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 1  |       25         |
        |      1       |   1           |     2            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 2  |       26         |
        |      1       |   1           |     3            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 3  |       27         |
        |      1       |   1           |     4            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 4  |       28         |
        |      1       |   1           |     5            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 5  |       29         |

        |      1       |   2           |     0            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 0  |       30         |
        |      1       |   2           |     1            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 1  |       31         |
        |      1       |   2           |     2            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 2  |       32         |
        |      1       |   2           |     3            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 3  |       33         |
        |      1       |   2           |     4            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 4  |       34         |
        |      1       |   2           |     5            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 5  |       35         |

## Global ThreadID Calculation:
   ---------------------------   

   We use the  Global BlockId & Local ThreadId to compute Global ThreadId

   (blockId * blockDim.x * blockDim.y * blockDim.z) + local ThreadId


        |   GlobalBlockID   |   LocalThreadId  |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 36 + 0      |           0           |
        |       0           |   0 * 36 + 1      |           1           |
        |       0           |   0 * 36 + 2      |           2           |
        |       0           |   0 * 36 + 3      |           3           |
        |       0           |   0 * 36 + 4      |           4           |
        |       0           |   0 * 36 + 5      |           5           |
        |       0           |   0 * 36 + 6      |           6           |
        |       0           |   0 * 36 + 7      |           7           |
        |       0           |   0 * 36 + 8      |           8           |
        |       0           |   0 * 36 + 9      |           9           |
        |       0           |   0 * 36 + 10     |           10          |
        |       0           |   0 * 36 + 11     |           11          |
        |       0           |   0 * 36 + 12     |           12          |
        |       0           |   0 * 36 + 13     |           13          |
        |       0           |   0 * 36 + 14     |           14          |
        |       0           |   0 * 36 + 15     |           15          |
        |       0           |   0 * 36 + 16     |           16          |
        |       0           |   0 * 36 + 17     |           17          |
        |       0           |   0 * 36 + 18     |           18          |
        |       0           |   0 * 36 + 19     |           19          |

        |       0           |   0 * 36 + 34     |           34          |
        |       0           |   0 * 36 + 35     |           35          |    

        |       1           |   1 * 36 + 0      |           36          |
        |       1           |   1 * 36 + 1      |           37          |
        |       1           |   1 * 36 + 2      |           38          |
        |       1           |   1 * 36 + 3      |           39          |
        |       1           |   1 * 36 + 4      |           40          |
        |       1           |   1 * 36 + 5      |           41          |
        |       1           |   1 * 36 + 6      |           42          |
        |       1           |   1 * 36 + 7      |           43          |
        |       1           |   1 * 36 + 8      |           44          |
        |       1           |   1 * 36 + 9      |           45          |
        |       1           |   1 * 36 + 10     |           46          |
        |       1           |   1 * 36 + 11     |           47          |
        |       1           |   1 * 36 + 12     |           48          |
        |       1           |   1 * 36 + 13     |           49          |
        |       1           |   1 * 36 + 14     |           50          |
        |       1           |   1 * 36 + 15     |           51          |
        |       1           |   1 * 36 + 16     |           52          |
        |       1           |   1 * 36 + 17     |           53          |
        |       1           |   1 * 36 + 18     |           54          |
        |       1           |   1 * 36 + 19     |           55          |

        |       0           |   1 * 36 + 34     |           70          |
        |       0           |   1 * 36 + 35     |           71          | 

         ---------------
       
        |       5           |   5 * 36 + 0      |           180         |
        |       5           |   5 * 36 + 1      |           181         |
        |       5           |   5 * 36 + 2      |           182         |
        |       5           |   5 * 36 + 3      |           183         |
        |       5           |   5 * 36 + 4      |           184         |
        |       5           |   5 * 36 + 5      |           185         |
        |       5           |   5 * 36 + 6      |           186         |
        |       5           |   5 * 36 + 7      |           187         |
        |       5           |   5 * 36 + 8      |           188         |
        |       5           |   5 * 36 + 9      |           189         |
        |       5           |   5 * 36 + 10     |           190         |
        |       5           |   5 * 36 + 11     |           191         |
        |       5           |   5 * 36 + 12     |           192         |
        |       5           |   5 * 36 + 13     |           193         |
        |       5           |   5 * 36 + 14     |           194         |
        |       5           |   5 * 36 + 15     |           195         |
        |       5           |   5 * 36 + 16     |           196         |
        |       5           |   5 * 36 + 17     |           197         |
        |       5           |   5 * 36 + 18     |           198         |
        |       5           |   5 * 36 + 19     |           199         |

        |       5           |   5 * 36 + 34     |           214         |
        |       5           |   5 * 36 + 35     |           215         | 




# Thread Indexing - 3D Grid and 3D Block

dim3 gridDim(4,3,2) = (x,y,z)
dim3 blockDim(6,3,2)  = (x,y,z)

gridDim.x = 4
gridDim.y = 3
gridDim.z = 2

blockDim.x = 6
blockDim.y = 3
blockDim.z = 2


Number of Blocks = 4 * 3 * 2 = 24
Nubmer of Thread per Block = 6 * 3 * 2 = 36
**Total Number of Threads  = 24 * 36 = 864**

## Grid Layout
----------------
Grid Comprises of 2 matrices of (3*4) , which can be represented as below.

Total Number of Blocks = 4 * 3 * 2 = 24 Blocks

    grid = [

    [  # z = 0
        [ Block(0,0,0), Block(0,0,1), Block(0,0,2), Block(0,0,3) ],  # y = 0
        [ Block(0,1,0), Block(0,1,1), Block(0,1,2), Block(0,1,3) ],  # y = 1
        [ Block(0,2,0), Block(0,2,1), Block(0,2,2), Block(0,2,3) ]   # y = 2
    ],
    [  # z = 1
        [ Block(1,0,0), Block(1,0,1), Block(1,0,2), Block(1,0,3)  ],  # y = 0
        [ Block(1,1,0), Block(1,1,1), Block(1,1,2), Block(1,1,3)  ],  # y = 1
        [ Block(1,2,0), Block(1,2,1), Block(1,2,2), Block(1,2,3)  ]  # y = 2
    ]
    
    ]

A unique global block ID is computed as: ``` blockId = ( blockIdx.z * gridDim.y * gridDim.x ) + ( blockIdx.y * gridDim.x ) + blockIdx.x ```




        | blockIdx.z   |   blockIdx.y  |  blockIdx.x      |      Calculation                  |  Global BlockId  |
        |------------- |---------------|------------------|-----------------------------------|------------------|
        |      0       |   0           |     0            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 0  |       0          |
        |      0       |   0           |     1            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 1  |       1          |
        |      0       |   0           |     2            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 2  |       2          |
        |      0       |   0           |     3            |    ( 0 * 3 * 4 ) + ( 0 * 4 ) + 3  |       3          |
        |      0       |   1           |     0            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 0  |       4          |
        |      0       |   1           |     1            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 1  |       5          |
        |      0       |   1           |     2            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 2  |       6          |
        |      0       |   1           |     3            |    ( 0 * 3 * 4 ) + ( 1 * 4 ) + 3  |       7          |
        |      0       |   2           |     0            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 0  |       8          |
        |      0       |   2           |     1            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 1  |       9          |
        |      0       |   2           |     2            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 2  |       10         |
        |      0       |   2           |     3            |    ( 0 * 3 * 4 ) + ( 2 * 4 ) + 3  |       11         |

        |      1       |   0           |     0            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 0  |       12          |
        |      1       |   0           |     1            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 1  |       13          |
        |      1       |   0           |     2            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 2  |       14          |
        |      1       |   0           |     3            |    ( 1 * 3 * 4 ) + ( 0 * 4 ) + 3  |       15          |
        |      1       |   1           |     0            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 0  |       16          |
        |      1       |   1           |     1            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 1  |       17          |
        |      1       |   1           |     2            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 2  |       18          |
        |      1       |   1           |     3            |    ( 1 * 3 * 4 ) + ( 1 * 4 ) + 3  |       19          |
        |      1       |   2           |     0            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 0  |       20          |
        |      1       |   2           |     1            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 1  |       21          |
        |      1       |   2           |     2            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 2  |       22          |
        |      1       |   2           |     3            |    ( 1 * 3 * 4 ) + ( 2 * 4 ) + 3  |       23          |

## Local Thread Indexing:
----------------------
 
   [

        [  # z = 0
            [ Thread(0,0,0), Thread(0,0,1), Thread(0,0,2), Thread(0,0,3), Thread(0,0,4), Thread(0,0,5)  ],  # y = 0
            [ Thread(0,1,0), Thread(0,1,1), Thread(0,1,2), Thread(0,1,3), Thread(0,1,4), Thread(0,1,5)  ],  # y = 1
            [ Thread(0,2,0), Thread(0,2,1), Thread(0,2,2), Thread(0,2,3), Thread(0,2,4), Thread(0,2,5)  ]   # y = 2
        ],
        [  # z = 1
            [ Thread(1,0,0), Thread(1,0,1), Thread(1,0,2), Thread(1,0,3), Thread(1,0,4), Thread(1,0,5)   ],  # y = 0
            [ Thread(1,1,0), Thread(1,1,1), Thread(1,1,2), Thread(1,1,3), Thread(1,1,4), Thread(1,1,5)   ],  # y = 1
            [ Thread(1,2,0), Thread(1,2,1), Thread(1,2,2), Thread(1,2,3), Thread(1,2,4), Thread(1,2,5)   ]  # y = 2
        ]
    
    ]


Local ThreadID is computed as: ``` Local ThreadId = ( threadIdx.z * blockDim.y * blockDim.x ) + ( threadIdx.y * blockDim.x ) + threadIdx.x ```



        | threadIdx.z  |   threadIdx.y |  threadIdx.x     |      Calculation                  |  Local ThreadId  |
        |------------- |---------------|------------------|-----------------------------------|------------------|
        |      0       |   0           |     0            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 0  |       0          |
        |      0       |   0           |     1            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 1  |       1          |
        |      0       |   0           |     2            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 2  |       2          |
        |      0       |   0           |     3            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 3  |       3          |
        |      0       |   0           |     4            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 4  |       4          |
        |      0       |   0           |     5            |    ( 0 * 6 * 3 ) + ( 0 * 6 ) + 5  |       5          |

        |      0       |   1           |     0            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 0  |       6          |
        |      0       |   1           |     1            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 1  |       7          |
        |      0       |   1           |     2            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 2  |       8          |
        |      0       |   1           |     3            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 3  |       9          |
        |      0       |   1           |     4            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 4  |       10         |
        |      0       |   1           |     5            |    ( 0 * 6 * 3 ) + ( 1 * 6 ) + 5  |       11         |

        |      0       |   2           |     0            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 0  |       12         |
        |      0       |   2           |     1            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 1  |       13         |
        |      0       |   2           |     2            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 2  |       14         |
        |      0       |   2           |     3            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 3  |       15         |
        |      0       |   2           |     4            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 4  |       16         |
        |      0       |   2           |     5            |    ( 0 * 6 * 3 ) + ( 2 * 6 ) + 5  |       17         |


        |      1       |   0           |     0            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 0  |       18         |
        |      1       |   0           |     1            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 1  |       19         |
        |      1       |   0           |     2            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 2  |       20         |
        |      1       |   0           |     3            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 3  |       21         |
        |      1       |   0           |     4            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 4  |       22         |
        |      1       |   0           |     5            |    ( 1 * 6 * 3 ) + ( 0 * 6 ) + 5  |       23         |

        |      1       |   1           |     0            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 0  |       24         |
        |      1       |   1           |     1            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 1  |       25         |
        |      1       |   1           |     2            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 2  |       26         |
        |      1       |   1           |     3            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 3  |       27         |
        |      1       |   1           |     4            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 4  |       28         |
        |      1       |   1           |     5            |    ( 1 * 6 * 3 ) + ( 1 * 6 ) + 5  |       29         |

        |      1       |   2           |     0            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 0  |       30         |
        |      1       |   2           |     1            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 1  |       31         |
        |      1       |   2           |     2            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 2  |       32         |
        |      1       |   2           |     3            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 3  |       33         |
        |      1       |   2           |     4            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 4  |       34         |
        |      1       |   2           |     5            |    ( 1 * 6 * 3 ) + ( 2 * 6 ) + 5  |       35         |


## Global ThreadID Calculation:
   ---------------------------   

   We use the  Global BlockId & Local ThreadId to compute Global ThreadId

   (blockId * blockDim.x * blockDim.y * blockDim.z) + local ThreadId


        |   GlobalBlockID   |   LocalThreadId  |   GlobalThreadID      |
        |-------------------|------------------|-----------------------|
        |       0           |   0 * 36 + 0      |           0           |
        |       0           |   0 * 36 + 1      |           1           |
        |       0           |   0 * 36 + 2      |           2           |
        |       0           |   0 * 36 + 3      |           3           |
        |       0           |   0 * 36 + 4      |           4           |
        |       0           |   0 * 36 + 5      |           5           |
        |       0           |   0 * 36 + 6      |           6           |
        |       0           |   0 * 36 + 7      |           7           |
        |       0           |   0 * 36 + 8      |           8           |
        |       0           |   0 * 36 + 9      |           9           |
        |       0           |   0 * 36 + 10     |           10          |
        |       0           |   0 * 36 + 11     |           11          |
        |       0           |   0 * 36 + 12     |           12          |
        |       0           |   0 * 36 + 13     |           13          |
        |       0           |   0 * 36 + 14     |           14          |
        |       0           |   0 * 36 + 15     |           15          |
        |       0           |   0 * 36 + 16     |           16          |
        |       0           |   0 * 36 + 17     |           17          |
        |       0           |   0 * 36 + 18     |           18          |
        |       0           |   0 * 36 + 19     |           19          |

        |       0           |   0 * 36 + 34     |           34          |
        |       0           |   0 * 36 + 35     |           35          |    

        |       1           |   1 * 36 + 0      |           36          |
        |       1           |   1 * 36 + 1      |           37          |
        |       1           |   1 * 36 + 2      |           38          |
        |       1           |   1 * 36 + 3      |           39          |
        |       1           |   1 * 36 + 4      |           40          |
        |       1           |   1 * 36 + 5      |           41          |
        |       1           |   1 * 36 + 6      |           42          |
        |       1           |   1 * 36 + 7      |           43          |
        |       1           |   1 * 36 + 8      |           44          |
        |       1           |   1 * 36 + 9      |           45          |
        |       1           |   1 * 36 + 10     |           46          |
        |       1           |   1 * 36 + 11     |           47          |
        |       1           |   1 * 36 + 12     |           48          |
        |       1           |   1 * 36 + 13     |           49          |
        |       1           |   1 * 36 + 14     |           50          |
        |       1           |   1 * 36 + 15     |           51          |
        |       1           |   1 * 36 + 16     |           52          |
        |       1           |   1 * 36 + 17     |           53          |
        |       1           |   1 * 36 + 18     |           54          |
        |       1           |   1 * 36 + 19     |           55          |

        |       0           |   1 * 36 + 34     |           70          |
        |       0           |   1 * 36 + 35     |           71          | 

         ---------------
       
        |       23          |   23 * 36 + 0      |           828         |
        |       23          |   23 * 36 + 1      |           829         |
        |       23          |   23 * 36 + 2      |           830         |
        |       23          |   23 * 36 + 3      |           831         |
        |       23          |   23 * 36 + 4      |           832         |
        |       23          |   23 * 36 + 5      |           833         |
        |       23          |   23 * 36 + 6      |           834         |
        |       23          |   23 * 36 + 7      |           835         |
        |       23          |   23 * 36 + 8      |           836         |
        |       23          |   23 * 36 + 9      |           837         |
        |       23          |   23 * 36 + 10     |           838         |
        |       23          |   23 * 36 + 11     |           839         |
        |       23          |   23 * 36 + 12     |           840         |
        |       23          |   23 * 36 + 13     |           841         |
        |       23          |   23 * 36 + 14     |           842         |
        |       23          |   23 * 36 + 15     |           843         |
        |       23          |   23 * 36 + 16     |           844         |
        |       23          |   23 * 36 + 17     |           845         |
        |       23          |   23 * 36 + 18     |           846         |
        |       23          |   23 * 36 + 19     |           847         |

        |       23          |   23 * 36 + 34     |           862         |
        |       23          |   23 * 36 + 35     |           863         | 
