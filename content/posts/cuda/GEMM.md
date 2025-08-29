https://siboehm.com/articles/22/CUDA-MMM
https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
https://alvinwan.com/how-to-tile-matrix-multiplication/#how-to-sidestep-tiling-limits
https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/


https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview
https://www.youtube.com/watch?v=1E8GDR8QXKw&t=3033s


Phrases:- Tip of the Iceberg
          Threads across multiple Processors needs to be orchestrated in synchronization
          They way I like to look at it is
          I would like to underscore the importance of

Linear Regression
RNN 
CNN
Transformers
Matrix Factorization(Recommendation Systems)

Pre-Training
Post-Training
Inference

          

# Intro:-
Throughout the years AI/ML space has evolved from classic ML (Before 2010) => Deep Learning (2010-2020) => Generative AI (2016-Current), these evolutions at each phase are backed by different Archtiectures, there is one key aspect that keeps evovling along with these Architectures, starting from Predicting House Prices using Linear Regression, Sentiment Analysis of Product Review using Recurrent Neural Networks , finding right caption for our Insta Photos using Convolution Neural Networks  , Recommendation for movie(or) webseries on  Netflix , last but not least today's vibe coding & autonomous driving systems all of them have one maing ingredient.

By now all of us has access to plethora of LLM's ChatGPT,Gemini,Claude (technically these are interfaces) , and we know each LLM is like student being trained on the whole content on the internet ,  student has different phases of learning, pre-training (Preliminary Learning), Post Training(Training on the feedback recived) & Inference(Final Test) the GEMM is omnipresent.

While I was focusing on understanding the Tranformer Architecture and implementing it with Pytorch, I did not pay much attention to the internals of the functions being used, when I started exploring the internals peeling the layers of abstraction (Pytorch & Tensorflow) It was awakening for me to realize the thought process and research behind this, the curiosity triggered me to  explore the CUDA Architecture & Libraries (cuBLAS,cuDNN,CUTLASS) called by Pytorch functions.Below I tried to cover high level overview of GEMM Terminology and different flavours of GEMM and explain one of the approaches in detail , this is only the tip of the iceberg and the devil is in the details, I also provided the resources helped me to arrive at my current understanding , the resources/blogs also have further details about the different ways of GEMM.




# Matrix Multiplication:-
-----------------------
 Matrix Multiplication as we know has time complexity of O(N^3) , O(N^3) is not acceptable this definitely requires optimization, consider the scale of modern AI: a single attention mechanism in a large language model can involve large matrices and numerous matrix multipliations are involved across the layers of the models , If these core operations aren't efficient even the most performant inference engines like vLLM or Triton can't provide better results despite of the optimizations such as KV Cache, Prefill, Speculative Decoding etc...

## The Computational Challenge
Let's examine a concrete example. Consider multiplying two matrices: A (256×128) and B (128×256), resulting in matrix C (256×256).
To calculate any element C(i,j), we need to compute the dot product of row i from matrix A with column j from matrix B. For C(0,0), we multiply the first row of A with the first column of B and sum the results:

        ** C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0) + ... + A(0,127)*B(127,0 ) **

This requires 128 multiply-add operations. Since our result matrix has 256×256 = 65,536 elements, and each requires 128 operations, we need a total of 8,388,608 FMA (Fused Multiply-Add) operations.

## The Naive Sequential Approach

The most straightforward implementation uses three nested loops: This sequential approach calculates elements one by one: C(0,0) → C(0,1) → ... → C(255,255). On a modern CPU, this might take several seconds, which is unacceptable for real-time inference.

```

for i in range(256):      # For each row of A
    for j in range(256):  # For each column of B  
        C[i][j] = 0
        for k in range(128):  # Dot product computation
            C[i][j] += A[i][k] * B[k][j]

```

include visualization from Claude


## The Parallelization Opportunity

Here's the key thing to observe: each element of the result matrix C can be calculated independently. Computing C(255,255) doesn't depend on any previous calculations of matrix C. This independence is the foundation for parallelization.
Theoretically, we could spawn 65,536 threads—one for each result element. Thread-0 calculates C(0,0), Thread-1 calculates C(0,1), and so on. If we could execute all threads simultaneously, our computation time would be determined by the maximum time to perform 128 FMA operations rather than 8+ million.

** Analogy:- **  Manufacturing Plant that produces Cars, it requires the Skilled Technicians for assembling the Parts , the Technicians require Parts & Tools to assemble them as single Unit (Car)

Though I mentioned that we can parallelize the whole process by creating multiple threads one for each element, this has some constraints , to understand these constraints we need to understand GPU Architecture, in the next section we will try to understand minimum terminology that is required for us to understand.

** Analogy:- ** Distributed System are ubiquotous, whether we are using an E-commerce application, Listening to favourite Song on Spotify, Watching Podcast on Youtube all of them needs to scale along with the consumer requirements, these application would not be able to achieve this using single Database or Storgae Infrastructure since they can't scale beyond certain configuration (CPU, Memory,TCP/UDP Connections). Hence the application scale this using Multiple Systems of similar configuration that works together to achieve the objective of ensuring the users have good experience while using thier application.


# GPU Architecture: Processing and Memory Hierarchy

To understand why naive parallelization isn't straightforward, we need to examine GPU architecture. Like a manufacturing plant producing cars has critical resources like Technicians, Tools, Raw Materials &  and Place to keep all the required Parts. GPUs too have critical resources some of them are highlighted in below , while each NVIDIA GPU Architecture has lot of these configurations we are only touching few of the important configurations that are relevant in our context, we can find full architecture details for Ampere [here](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

## Processing Units (The Skilled Technicians):

**Streaming Multiprocessors (SMs):** The main production lines where work gets done, think of each SM as Workshop
**CUDA Cores/Tensor Cores:** Individual workstations in a Workshop
**Thread Blocks:** Teams of workers assigned to Complete a task
**Warps:** Small coordinated group. of 32 threads
**Threads:** Individual workers

## Memory Hierarchy (storage for parts and materials):

**Global RAM:** The main warehouse (large but distant)
**L2 Cache:** Regional storage - Shared across all SMs
**Shared Memory/L1 Cache:** Workshop floor storage (fast but limited)
**Registers:** immediate tools required for Technician (fastest, most limited)

## Resource Constraints:

Just as our manufacturing plant has limited skilled technicians and Shop Floor space, GPUs have finite resources. For NVIDIA's A100 (Ampere architecture), we have to be aware of these constrains before writing CUDA functions or kernels to achieve our objectives

**108 Streaming Multiprocessors**
**2,048 maximum threads per SM**
**64 maximum Warps per SM**
**1024 Threads per Block**
**64 FP32 cores + 32 FP64 cores + 4 Tensor cores per SM**
**Total Threads = 2048 Threads per SM * 108 SM = 221,184 Threads per GPU**

**Global (HBM) memory:** 40GB
**L2 Cache:** 40MB
**Shared-memory/L1(Shared across Thread Block):** 192KB
**Registers:** 64K 32-bit Registers per SM,  255 Registers per Thread 

# CUDA Programming Terminology:

## Kernel:      
 
A kernel is a function that executes on the GPU to achieve our computational objective. In our manufacturing analogy, it's the instruction manual that guides each technician on the steps to follow for their specific task.

For matrix multiplication, our kernel would contain the instructions for computing C[i][j] = A[i][k] * B[k][j] for assigned matrix elements.

Though the GPU performs the actual computation, the CPU orchestrates the entire process. The way I like to think about it is like maritime logistics where cargo is transferred to smaller vessels (lighters) before reaching the main ship: Before Kernel is launched on GPU, CPU copies the data to GPU HBM Memory(Global Memory)


   - Declare Host Variables (Host = CPU)
   - Declare Device Variables (Device = GPU)
   - Allocate Device Memory (Memory to Store the Data Required in Global Memory HBM)
   - Copy Host → Device (Transfer input data to GPU)
   - Launch Kernel (Execute Main Function)
   - Copy Device → Host (Retrieve results from GPU to CPU)
   - Free Memory(Clean up alloacted memory)

 ```
    __global__ void scaleMatrix(float* A,  int N , int C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (idx < N) {
           
            float sum = 0.0f;
            A[idx] = C * A[idx];
        }
    }

 ```


 ```

    // Example workflow
    float *h_A;          // Host arrays
    float *d_A;          // Device arrays

    // Allocate and initialize host memory
    h_A = (float*)malloc(size);
    cudaMalloc(&d_A, size);          // Allocate device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Copy to device

    // Launch kernel
    dim3 blockSize(256);
    dim3 gridSize(256);
    scaleMatrix<<<gridSize, blockSize>>>(d_A, N, C);

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost); // Copy results back

 ```

 ## SIMT :- Single Instruction Multiple Thread

SIMT is a Programming Model for writing CUDA Kernel's , consider multiplying Matrix A of size(256,256) by a constant C , the instruction A[i][j] = C * A[i][j]  remains same for all the iterations of the loop except the change in index, this nested loop can be converted to be executed in parallel , where each thread calculates one element in GPU.

Note:- In modern Architectures there are some modifications where collection of Threads work on single element like Thread Block Clusters, but for our discussion this is fine now.

   ```

    for i in range(256):
        for j in range(256):
            A[i][j] = C * A[i][j]

   ```

## Thread Hierarchy

To perform computations on the GPU, we write **kernels**, whether developing custom kernels or using existing ones, each kernel requires **processing units (CUDA/Tensor cores)** and **memory resources** for execution. These resources are orgaized according to configurations specified during kernel launch.

The resource allocation involves a two-tier responsibility model:

- Developer-controlled: Grid dimensions, Block dimensions, shared memory usage, Register Usate
- Hardware/scheduler-controlled: Warp Scheduling, Streaming Multiprocessor assignment, Instruction scheduling

Developer's doesn't specify individual threads directly. Instead number of required threads expressed through Grid & Thread Blocks configuration which uses a hierarchical approach: **Grid → Thread Blocks → Warps → Threads**, both Grids & Blocks configurations can be expressed using 1D,2D & 3D to address problem statements belonging to diffrent domains.

### Warp

A critical aspect of CUDA's execution model is **Warp** - This is not controlled by Developers, whiel we specify the Grid & Block dimension we don't specify the Warp which is controlled by Schedulers which is constant and always Group of 32 threads belonging to one unit.

An importatnt note about Warp is Even if we specify just 1 thread per Block, the warp scheduler still allocates a full warp of 32 threads, with 31 threads remaining unused. This significantly impacts SM utilization and should be considered when designing kernel launch configurations. Below table gives glimpse the impact of Thread Block size on Warp utilization.


| Thread Block Size | Warps Used  | Warp Utilization | 
|-------------------|-------------|------------------|
| 16 Threads        |    1 Warp   |    50% (16/32)   |
| 32 Threads        |    2 Warps  |    100% (32/32)  |
| 48 Threads        |    2 Warps  |    75% (48/64)   |
| 64 Threads        |    2 Warps  |    100% (64/64)  |
| 96 Threads        |    3 Warps  |    100% (96/96)  |


### Thread Block Configuration


```dim3 blockSize(16, 16) // 2D Thread Block```

- Each thread block contains 256 threads (16 × 16)
- 16 threads along the X dimension
- 16 threads along the Y dimension
- This creates a logical 2D arrangement of threads within each block

### Grid Configuration

```dim3 gridSize(16, 16);   // 2D grid of thread blocks```

- The grid contains 256 thread blocks (16 × 16)
- 16 thread blocks along the X dimension
- 16 thread blocks along the Y dimension
- This creates a logical 2D arrangement of thread blocks

### Kernel Launch

```kernel<<<gridSize, blockSize>>>(parameters)```

- Thread blocks per grid: 16 × 16 = 256 thread blocks
- Threads per block: 16 × 16 = 256 threads
- Warps per Block : 256 / 32 = 8 Warps per Block 
- Total Warps :  8 * 256  = 2048 Warps
- Total threads: 256 × 256 = 65,536 threads

This is one possible configuration for processing a 256×256 matrix where each thread handles one matrix element(This is not Optimal configuration)


### Logical vs Physical Representation

The above representation is logical representation , internally GPU operates on liner indexes, hence it would be converted to linear index, either the developer can do it or internally it is converted to linear index, below image gives idea on how to 

- Linear thread IDs within each warp (32 consecutive threads)
- Linear memory addresses in global memory
- Sequential instruction execution within warps

thread Indexes are linear hence while accessing the elements of matrix we can use either of the representations either 2D or linear indexing below is one example of how 2D indexes can be converted to linear indexing.

![2D_1D_Index_Mapping](/images/cuda/gemm/2D_1D_Index_Mapping.png)  

![3D_1D_Index_Mapping](/images/cuda/gemm/3D_1D_Index_Mapping.png)



## Memory Coalescing : Efficient Data Access Patterns

The way matrix elements are stored in memory significantly impacts kernel performance.  Understanding how threads access memory  is crucial for achieving optimal GPU performance in matrix operations to avoid Kernels starving for data due to slow memory access.
  
### GPU Memory Operations Overview

GPU memory operations can be categorized into three phases:

1. Load  - Transfer data from global memory to registers directly or via shared memory
2. Compute -  Execute the actual mathematical operations on data in registers
3. Store operations - Write results back to global memory

The important point to consider is that memory bandwidth not computational power which often becomes the bottleneck in GPU kernels. This is why optimizing memory access patterns has significant role while writing custom CUDA Kernels.

Memory coalescing occurs when threads within the same warp access consecutive memory addresses. When this happens, the GPU can combine multiple memory requests into fewer, wider memory transactions, dramatically improving bandwidth utilization.

Note:- Coalescing is determined by the access pattern of the 32 threads within a single warp, not by the overall grid structure.

### Memory Storage Layouts

By default, C++ stores matrices in **row-major layout** while Fortran uses **column-major layout**. This is analogous to how databases can be organized as row-oriented (traditional RDBMS) or column-oriented (analytical databases) based on storage patterns.

![Row Column Layout](/images/cuda/gemm/Row_Column_Layout.png)

  
Whether we are using CPU or GPU for computation, data first needs to be brought to execution units  ALU in CPU  or CUDA Cores in GPU which is accomplished using Load instruction , Upon load instruction data is moved from Global Memory to Registers either directly or through Shared Memory , Registers are High speed memory . to keep it simple for our discussion each Load instruction can only access certain number of bytes in one cycle, when all of our threads in a Warp are accessing adjacent locations one load instruction can fetch all the inputs required for a threads in Warp to be processed , If the threads of the same Warp are accessing the elements that are distant that leads to multiple load instructions from Global Memory which is time taking process as transfer of data from Global Memory is constrained by the bandwidth , because CUDA cores complete the instructions and waits for the data while the load instruction still fetching the data from Global Memory, hence it is paramount important to make sure the data should be arranged such that threads in a warp access consectutive elements for thier respective operations.

### Manufacturing Analogy:

In our Manufacturing analogy, if all the technicians are working on same part of Car, all the technicians require parts from same bin or near by Bins , one supply truck make a single trip to to load all parts from consecutive bins, otherwise if each technician is working on different part of car which requires part from different bins  this makes it difficult to load all parts from different bins , this might require multiple iterations to bring parts required for all technicians , which impacts the overall execution of the manufacturing process

Let us go through some examples of Coalesce and Non-Coalesce patterns to understand this better

### Coalesced Memory Access Pattern Example

Consider matrix multiplication for matrices **A=(1024×1024)**, **B=(1024×1024)**, **C=(1024×1024)** using a configuration that results in **optimal coalesced memory access**.

#### Configuration Parameters

```
gridDim = (32,1024)  => 32 Thread Blocks in X dimension, 1024 Thread Blocks in Y dimension
blockDim = (32,1)    => 32 Threads in X dimension, 1 Thread in Y dimension

gridDim.x = 32       blockDim.x = 32
gridDim.y = 1024     blockDim.y = 1

threadIdx.x range: 0-31     blockIdx.x range: 0-31
threadIdx.y range: 0        blockIdx.y range: 0-1023
```

**Total Configuration:**
- **Total Thread Blocks:** 32 × 1024 = 32,768 blocks
- **Total Threads per Block:** 32 × 1 = 32 threads
- **Total Warps per Block:** 32 ÷ 32 = 1 warp per block
- **Total Threads:** 1,048,576 threads (perfect for 1024×1024 matrix)

#### Thread Block Organization

```
Thread Block Layout (32×1024 grid):

Thread Block (0,0)     Thread Block (1,0)     Thread Block (2,0)     ...     Thread Block (31,0)
Thread Block (0,1)     Thread Block (1,1)     Thread Block (2,1)     ...     Thread Block (31,1)
Thread Block (0,2)     Thread Block (1,2)     Thread Block (2,2)     ...     Thread Block (31,2)
.
.
.
Thread Block (0,1023)  Thread Block (1,1023)  Thread Block (2,1023)  ...     Thread Block (31,1023)
```

#### Thread Organization within Each Block

```
Thread Layout (32×1 per block):

threadIdx(0,0)    threadIdx(1,0)    threadIdx(2,0)    ...    threadIdx(31,0)
```

**Key Insight:** Each block contains exactly **32 threads = 1 complete warp**

#### Linear Index Calculation Formula

```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
idx = row * matrix_width + col    (where matrix_width = 1024)
```

#### Coalesced Memory Access Analysis

##### Perfect Coalescing: Each Warp Accesses Consecutive Memory

With `blockDim(32,1)`, each warp contains 32 threads that access **32 consecutive memory locations** within the same matrix row.

##### Thread Block (0,0) - Complete Warp Analysis:

**Thread 0:**
```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 0 * 32 + 0 = 0
idx = 0 * 1024 + 0 = 0
```

**Thread 1:**
```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=1, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 0 * 32 + 1 = 1
idx = 0 * 1024 + 1 = 1
```

**Thread 2:**
```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=2, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 0 * 32 + 2 = 2
idx = 0 * 1024 + 2 = 2
```

**...continuing pattern...**

**Thread 31:**
```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=31, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 0 * 32 + 31 = 31
idx = 0 * 1024 + 31 = 31
```

**✅ Perfect Coalescing:** All 32 threads access consecutive indices **0, 1, 2, ..., 31**

##### Thread Block (1,0) - Next Consecutive Block:

**Thread 0:**
```
Parameters: blockIdx.x=1, blockIdx.y=0, threadIdx.x=0, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 1 * 32 + 0 = 32
idx = 0 * 1024 + 32 = 32
```

**Thread 1:**
```
Parameters: blockIdx.x=1, blockIdx.y=0, threadIdx.x=1, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 1 * 32 + 1 = 33
idx = 0 * 1024 + 33 = 33
```

**...continuing pattern...**

**Thread 31:**
```
Parameters: blockIdx.x=1, blockIdx.y=0, threadIdx.x=31, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 1 * 32 + 31 = 63
idx = 0 * 1024 + 63 = 63
```

**✅ Perfect Coalescing:** All 32 threads access consecutive indices **32, 33, 34, ..., 63**

##### Thread Block (31,0) - Final Block in Row 0:

**Thread 0:**
```
Parameters: blockIdx.x=31, blockIdx.y=0, threadIdx.x=0, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 31 * 32 + 0 = 992
idx = 0 * 1024 + 992 = 992
```

**Thread 1:**
```
Parameters: blockIdx.x=31, blockIdx.y=0, threadIdx.x=1, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 31 * 32 + 1 = 993
idx = 0 * 1024 + 993 = 993
```

**...continuing pattern...**

**Thread 31:**
```
Parameters: blockIdx.x=31, blockIdx.y=0, threadIdx.x=31, threadIdx.y=0

Calculation:
row = 0 * 1 + 0 = 0
col = 31 * 32 + 31 = 1023
idx = 0 * 1024 + 1023 = 1023
```

**✅ Perfect Coalescing:** All 32 threads access consecutive indices **992, 993, 994, ..., 1023**

#### Summary: Coalesced Access Pattern

| Thread Block | Memory Indices Accessed | Pattern | Gap Size |
|--------------|------------------------|---------|----------|
| Block (0,0)  | 0, 1, 2, ..., 31      | ✅ Consecutive | 0 |
| Block (1,0)  | 32, 33, 34, ..., 63   | ✅ Consecutive | 0 |
| Block (2,0)  | 64, 65, 66, ..., 95   | ✅ Consecutive | 0 |
| ...          | ...                    | ✅ Consecutive | 0 |
| Block (31,0) | 992, 993, 994, ..., 1023 | ✅ Consecutive | 0 |

#### Key Advantages

1. **No Memory Gaps:** Every thread in a warp accesses consecutive memory addresses
2. **Single Memory Transaction:** Each warp requires only 1 memory transaction
3. **Optimal Bandwidth Utilization:** 100% of memory bandwidth is effectively used
4. **Maximum Performance:** Optimal memory access pattern for GPU architecture

#### Complete Row Coverage

The configuration provides perfect coverage of matrix row 0:
- **32 thread blocks** × **32 threads per block** = **1,024 threads**
- **Coverage:** Indices 0 through 1,023 (complete row)
- **No overlap or gaps:** Each index is processed by exactly one thread

#### Warp-Level Analysis

Since each thread block contains exactly **32 threads = 1 warp**:

| Thread Block | Warp | Threads | Memory Access Pattern |
|--------------|------|---------|---------------------|
| Block (0,0) | Warp 0 | 0-31 | Consecutive: 0-31 ✅ |
| Block (1,0) | Warp 0 | 0-31 | Consecutive: 32-63 ✅ |
| Block (2,0) | Warp 0 | 0-31 | Consecutive: 64-95 ✅ |
| ... | ... | ... | ... |
| Block (31,0) | Warp 0 | 0-31 | Consecutive: 992-1023 ✅ |

**Result:** Every warp achieves perfect memory coalescing!

#### Why This Configuration Works

1. **blockDim.y = 1:** Ensures all threads in a warp stay within the same matrix row
2. **blockDim.x = 32:** Matches warp size, providing optimal thread-to-warp mapping
3. **Consecutive Access:** ThreadIdx.x directly maps to consecutive column indices
4. **No Row Spanning:** Warps never span multiple matrix rows, eliminating gaps

**Key Insight:** The fundamental success of `blockDim(32,1)` is that it aligns perfectly with GPU warp architecture, ensuring each warp accesses a single contiguous block of memory within the same matrix row.


### Non-Coalesced Memory Access Pattern Example

Consider matrix multiplication for matrices **A=(1024×1024)**, **B=(1024×1024)**, **C=(1024×1024)** using a configuration that results in **non-coalesced memory access**.

#### Configuration Parameters

```
gridDim = (64,64)   => 64 Thread Blocks in X dimension, 64 Thread Blocks in Y dimension
blockDim = (16,16)  => 16 Threads in X dimension, 16 Threads in Y dimension

gridDim.x = 64      blockDim.x = 16
gridDim.y = 64      blockDim.y = 16

threadIdx.x range: 0-15    blockIdx.x range: 0-63
threadIdx.y range: 0-15    blockIdx.y range: 0-63
```

**Total Configuration:**
- **Total Thread Blocks:** 64 × 64 = 4,096 blocks
- **Total Threads per Block:** 16 × 16 = 256 threads
- **Total Warps per Block:** 256 ÷ 32 = 8 warps
- **Total Threads:** 1,048,576 threads (perfect for 1024×1024 matrix)

#### Thread Block Organization

```
Thread Block Layout (64×64 grid):

Thread Block (0,0)    Thread Block (1,0)    Thread Block (2,0)    ...    Thread Block (63,0)
Thread Block (0,1)    Thread Block (1,1)    Thread Block (2,1)    ...    Thread Block (63,1)
Thread Block (0,2)    Thread Block (1,2)    Thread Block (2,2)    ...    Thread Block (63,2)
.
.
.
Thread Block (0,63)   Thread Block (1,63)   Thread Block (2,63)   ...    Thread Block (63,63)
```

#### Thread Organization within Each Block

```
Thread Layout (16×16 per block):

threadIdx(0,0)    threadIdx(1,0)    threadIdx(2,0)    ...    threadIdx(15,0)
threadIdx(0,1)    threadIdx(1,1)    threadIdx(2,1)    ...    threadIdx(15,1)
threadIdx(0,2)    threadIdx(1,2)    threadIdx(2,2)    ...    threadIdx(15,2)
.
.
.
threadIdx(0,15)   threadIdx(1,15)   threadIdx(2,15)   ...    threadIdx(15,15)
```

#### Linear Index Calculation Formula

```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
idx = row * matrix_width + col    (where matrix_width = 1024)
```

#### Non-Coalesced Memory Access Analysis

##### Problem: Warps Span Multiple Matrix Rows

With `blockDim(16,16)`, each warp contains 32 consecutive threads that span **2 matrix rows**, creating large memory gaps.

##### Warp 0 (Threads 0-31) in Thread Block (0,0):

**First 16 threads (threadIdx.y = 0):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=0

Calculation:
row = 0 * 16 + 0 = 0
col = 0 * 16 + (0-15) = 0-15
idx = 0 * 1024 + (0-15) = 0-15

Memory indices accessed: 0, 1, 2, 3, ..., 14, 15
```

**Next 16 threads (threadIdx.y = 1):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=1

Calculation:
row = 0 * 16 + 1 = 1
col = 0 * 16 + (0-15) = 0-15
idx = 1 * 1024 + (0-15) = 1024-1039

Memory indices accessed: 1024, 1025, 1026, ..., 1038, 1039
```

**❌ Gap Analysis:** **1009 elements** between indices 15 and 1024!

##### Warp 1 (Threads 32-63) in Thread Block (0,0):

**Threads 32-47 (threadIdx.y = 2):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=2

Calculation:
row = 0 * 16 + 2 = 2
col = 0 * 16 + (0-15) = 0-15
idx = 2 * 1024 + (0-15) = 2048-2063

Memory indices accessed: 2048, 2049, 2050, ..., 2062, 2063
```

**Threads 48-63 (threadIdx.y = 3):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=3

Calculation:
row = 0 * 16 + 3 = 3
col = 0 * 16 + (0-15) = 0-15
idx = 3 * 1024 + (0-15) = 3072-3087

Memory indices accessed: 3072, 3073, 3074, ..., 3086, 3087
```

**❌ Same Gap Problem:** **1009 elements** between indices 2063 and 3072!

##### Final Warp Example - Warp 7 (Threads 224-255):

**Threads 224-239 (threadIdx.y = 14):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=14

Calculation:
row = 0 * 16 + 14 = 14
col = 0 * 16 + (0-15) = 0-15
idx = 14 * 1024 + (0-15) = 14336-14351
```

**Threads 240-255 (threadIdx.y = 15):**

```
Parameters: blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=15

Calculation:
row = 0 * 16 + 15 = 15
col = 0 * 16 + (0-15) = 0-15
idx = 15 * 1024 + (0-15) = 15360-15375
```

#### Summary: Non-Coalesced Access Pattern

| Warp | First 16 Threads Access | Next 16 Threads Access | Gap Size | Coalesced? |
|------|------------------------|------------------------|----------|------------|
| Warp 0 | 0-15 (row 0) | 1024-1039 (row 1) | 1009 elements | ❌ No |
| Warp 1 | 2048-2063 (row 2) | 3072-3087 (row 3) | 1009 elements | ❌ No |
| Warp 2 | 4096-4111 (row 4) | 5120-5135 (row 5) | 1009 elements | ❌ No |
| ... | ... | ... | 1009 elements | ❌ No |
| Warp 7 | 14336-14351 (row 14) | 15360-15375 (row 15) | 1009 elements | ❌ No |

#### Critical Problems

1. **Large Memory Gaps:** Every warp has a 1009-element gap between consecutive thread accesses
2. **Multiple Memory Transactions:** Each warp requires 2-4 separate memory transactions instead of 1
3. **Poor Bandwidth Utilization:** Only 25-50% of memory bandwidth is effectively used
4. **Performance Degradation:** 2-4x slower compared to coalesced access patterns

#### Solution

Use **`blockDim(32,1)`** or **`blockDim(32,32)`** configurations where:
- Each warp accesses **32 consecutive memory locations** within the same matrix row
- **No gaps** between thread accesses within a warp
- **Single memory transaction** per warp
- **100% bandwidth utilization**

**Key Insight:** The fundamental issue with `blockDim(16,16)` is that it forces warps to span multiple matrix rows, breaking the consecutive memory access pattern required for optimal coalescing.


### Coalescing in the context of GEMM

 when we examine the Matrix multiplication , though we are able to achieve parallelization by using Grid & Thread Block hierarchy , memory bandwidth causes issues as access to one of our matrix is not coalesced , consider calculating element C[0][0] from multiplying matrices A(256,128) & B(128 , 256) . Observing closely the calculation for C[0][0] we can see that access pattern to matrix A is coalesced while the access patterns to matrix B is not coalesced, this leads us to explore further optimiztions for GEMM.

```
 C[i][j] = A[i][k] * B[k][j]
 C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0] + ......+ A[0][127] * B[127][0]

```
    


To handle our 65,536 threads, we need at least 32 SMs (65,536 ÷ 2,048 ≈ 32). While the A100 has enough SMs, the constraint isn't just processing units—it's also memory bandwidth and access patterns.

 to state the obvious here we can notice two constraints, technicians who are skilled technicians, generally a plant only has few of them, On the shopfloor they have finite place to keep the parts required to assemble the car, though the parts are available in the warehouse they can't keep all the parts on the Shopfloor due to space constraints.This is similar to our Memory constraints in GPU 


Resultant Matrix has 256 * 256 elements C(0,0) => C(256,256) for each elements it requires 128 Operations for instance to calculate the element at C(0,0) we multiply the first row of A matrix A(0, *) with first column of B (*,0) and perform addition operation among all the intermediate outputs , in both cases the * represents 128 elements, hence for calculating C(0,0) it requries  128 Operations, applying this to a matrix of C it requires 256*256*128 operations, these operations are termed as FMA (Fused Multiply & Add) in CUDA terminology.

This can be calculated using different approaches, the naive approache is to calculate each element in sequential manner , C(0,0)=>C(0,1) ... =>C(256,256) this is time taking process, we observe that there is no depednecy between each of these calculations meaning to calculate C(256,256) we don't depend on any of the previous calculations of C matrix hence each element can be calculated independently this is the basic premise that allows us to think towards parallelizing the matrix multiplication operation, to parallelize the whole operation we would need 65536 (256*256) Threads where each thread is responsible for calculating one element in result matrix for instance Thread-0 calculates C(0,0), Thread-65536 Calculates C(255,255). Like mentioned earlier to allocate this work to 65536 threads there are few limitations on the processing & memory fronts below are few
important specifications for Ampere Architecture(A100). Considering the outlined specification we would need to distribute 65536 across different Streaming Multiprocessors we would need 32 SM's (65536/2048).




Ampere Architecture :- 
--------------
108 Streaming Multiprocessors
6912 CUDA Cores (64 CUDA Cores per SM)
432 Tensor Cores (4 Tensor Cores per SM)
Max Threads per SM - 2048
Max Warps per SM - 64
Max ThreadBlock Size:- 1024




CUDA Terminilogy:-
------------
 
Like described earlier GPU has two pillars that whole execution is built on top of  executing Kernels Processing Hierarychy  & Memory Hierarchy
 
 Kernel :- Function achieving help us to achieve certain objective, ex:- Multiplying matrix by a constant
 Grids:- Collection of Thread Blocks
 Thread Blocks:- Group of Warps
 Warps:- Groupd of Threads
 Threads :- Smallest Unit of execution framework


 Memory Hierarchy
 ------
 Global Memory :- DRAM
 L2 Cache
 Shared Memory /L1 Cache
 Registers




 GEMM has different flavours depending on precision of Operands being used , Processing Units (CUDA Cores, Tensor Cores)

  C = A@B

 DGEMM => Double Precision GEMM (Datatype of Operands - Float64)
 SGEMM => Single Precision GEMM (Datatype of Operands - Float32)
 HGEMM => Half Precision GEMM (Datatype of Operands - Float16)
 IGEMM -> Implicit GEMM 
 
 Strided Batched GEMM => Batch Multiple Small GEMM
 Grouped GEMM => 


TILED GEMM:-
-------

Outro:-
------


Though I started my ML journey a while ago I did not pay much attention to this important aspect that touches every algorithm and Architecture, 


https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0/

https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/

https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/
https://developer.nvidia.com/blog/advanced-optimization-strategies-for-llm-training-on-nvidia-grace-hopper/
https://developer.nvidia.com/blog/profiling-llm-training-workflows-on-nvidia-grace-hopper/


https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu