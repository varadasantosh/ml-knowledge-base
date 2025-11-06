---
title: "Tiled GEMM"
---

<!--more-->

# GEMM with Shared Memory Tiling

## Recap: Naive GEMM Limitations
In our previous exploration of naive GEMM implementation, we discovered that despite achieving optimal memory access patterns—broadcast reads for Matrix A and coalesced reads for Matrix B—performance remained severely limited at approximately 2% of theoretical peak.

The root cause was poor data reuse due to cache capacity constraints. While our (32,32) block configuration created efficient warp-level memory patterns, frequent cache evictions forced repeated global memory accesses for the same data elements. This resulted in a memory-bound kernel unable to fully utilize the GPU's computational resources.

## Moving Forward: Shared Memory as the Solution

To overcome these limitations of naive GEMM, we need explicit control over data locality and reuse. The L1 cache, being a hardware-managed cache controlled by the execution framework, provides no guarantee that data brought into cache will remain available for subsequent transactions.

Shared memory, managed by the programmer and often called a "software cache," provides a solution by allowing programmers to control data movement explicitly. Once data is loaded into shared memory, it is guaranteed to remain available until either overwritten by the program or the kernel execution completes.

However, it is not possible to bring all matrix data into shared memory simultaneously. To understand why, let us examine the key limitations that prevent this approach:



## CUDA Programming Constraints - Ampere Architecture

Understanding memory and thread limitations when developing CUDA applications for matrix operations.



<style>
/* Custom CSS for CUDA Constraints Documentation */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

.constraint-container {
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 20px 0;
}

.constraint-section {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    margin: 20px 0;
    overflow: hidden;
}

.section-header {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    color: white;
    padding: 16px 50px;
    font-weight: 600;
    font-size: 18px;
    margin: 0;
}

.section-content {
    padding: 20px;
    background: #fafafa;
}

.constraint-item {
    margin-bottom: 15px;
    padding: 15px;
    background: white;
    border-left: 4px solid #4CAF50;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.constraint-item h4 {
    margin: 0 0 10px 0;
    color: #2E7D32;
    font-size: 16px;
    font-weight: 600;
}

.constraint-item p {
    margin: 0;
    color: #555;
    font-size: 14px;
}

.calculation {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    color: #333;
    margin: 10px 0;
    display: block;
}

.highlight {
    background: #E8F5E8;
    color: #2E7D32;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.warning-box {
    background: #FFF3E0;
    border: 1px solid #FFB74D;
    border-left: 4px solid #FF9800;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
}

.warning-title {
    font-weight: 600;
    color: #E65100;
    margin-bottom: 8px;
    font-size: 14px;
}

.warning-box p {
    margin: 0;
    color: #BF360C;
}

.percentage {
    color: #D32F2F;
    font-weight: 700;
    font-size: 16px;
}

.memory-req {
    background: #E3F2FD;
    border-left: 4px solid #2196F3;
    padding: 12px;
    margin: 10px 0;
    border-radius: 4px;
}

.thread-analysis {
    background: #F3E5F5;
    border-left: 4px solid #9C27B0;
    padding: 12px;
    margin: 10px 0;
    border-radius: 4px;
}
</style>

## <span class="section-header">Shared Memory Constraints</span>

<div class="constraint-item">
<h4>Memory Allocation Limits</h4>
<p>Shared memory is configurable, with a maximum allocation of <span class="highlight">164 KB per thread block</span> 
(from the total 192 KB block)</p>


<div class="constraint-item">
<h4>Example Matrix Memory Requirements</h4>
<p>Our example matrix A (256×256) requires:</p>

<div class="calculation">256 × 256 × 4 bytes = 256 KB of memory</div>
</div>

<div class="constraint-item">
<h4>Access Restrictions</h4>
<p>Data in shared memory is accessible only to threads within the same thread block</p>
</div>

<div class="warning-box">
<div class="warning-title">Memory Limitation Impact</div>
<p>The matrix requires 256 KB but only 164 KB is available per block, creating a significant constraint for large matrix operations.</p>
</div>

---

## <span class="section-header">Thread Block Limitations</span>

<div class="constraint-item">
<h4>Total Threads Required</h4>
<p>To calculate the complete output matrix C (256×256 = <span class="highlight">65,536 elements</span>), we would need 65,536 threads</p>
</div>

<div class="constraint-item">
<h4>Ampere Architecture Limits</h4>
<p>Maximum threads per thread block in Ampere architecture: <span class="highlight">1,024 threads</span></p>
</div>

<div class="constraint-item">
<h4>Thread Capacity Analysis</h4>
<p>This represents only approximately <span class="percentage">1.6%</span> of the required threads:</p>

<div class="calculation">1,024 ÷ 65,536 = 0.0156 (1.56%)</div>
</div>

<div class="warning-box">
<div class="warning-title">Threading Bottleneck</div>
<p>The huge gap between required threads (65,536) and available threads per block (1,024) necessitates careful work distribution across multiple thread blocks.</p>
</div></div>

## Tiling Strategy Options for 256×256 Matrices

These constraints reveal that while matrix A cannot fit entirely into one thread block's shared memory, it can be split into smaller sub-matrices (tiles) that do fit. This approach forms the foundation of Tiled GEMM, where matrices are divided into manageable tiles that individual thread blocks can process effectively using shared memory. The tiling strategy is designed from the output matrix's perspective.


<style>
/* Custom CSS for Tiling Documentation */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

.tiling-section {
    background: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 20px;
    margin: 20px 0;
    border-radius: 4px;
}

.tiling-options {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.tile-card {
    background: white;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    transition: all 0.3s ease;
}

.tile-card:hover {
    border-color: #4CAF50;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
}

.tile-size {
    font-size: 18px;
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 8px;
}

.tile-count {
    font-size: 20px;
    font-weight: 700;
    color: #1976d2;
    margin: 8px 0;
}

.memory-usage {
    font-size: 12px;
    color: #666;
    background: #f5f5f5;
    padding: 4px 8px;
    border-radius: 12px;
    display: inline-block;
}

.comparison-table {
    width: 90%;
    max-width: 500px;
    border-collapse: collapse;
    margin: 25px auto 25px 0;
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    border: 1px solid #e0e0e0;
}

.comparison-table th {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    color: white;
    padding: 20px 24px;
    text-align: center;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: 0.3px;
    border: none;
    white-space: nowrap;
}

.comparison-table td {
    padding: 20px 24px;
    text-align: center;
    border: none;
    border-bottom: 1px solid #f0f0f0;
    font-size: 16px;
    color: #333;
    font-weight: 500;
    min-width: 80px;
}

/* Remove border from last row */
.comparison-table tbody tr:last-child td {
    border-bottom: none;
}

.comparison-table tbody tr {
    background: white;
    transition: all 0.2s ease;
}

.comparison-table tbody tr:hover {
    background: #f8fffe;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
}

/* First column (Tile Size) styling */
.comparison-table td:first-child {
    font-weight: 700;
    color: #2e7d32;
    font-size: 17px;
    background: #f8f9fa;
    border-right: 1px solid #e8f5e8;
}

/* Better column widths */
.comparison-table th:first-child,
.comparison-table td:first-child {
    width: 20%;
}

.comparison-table th:nth-child(2),
.comparison-table td:nth-child(2) {
    width: 25%;
}

.comparison-table th:nth-child(3),
.comparison-table td:nth-child(3) {
    width: 25%;
}

.comparison-table th:nth-child(4),
.comparison-table td:nth-child(4) {
    width: 30%;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .comparison-table {
        width: 95%;
        font-size: 14px;
    }
    
    .comparison-table th {
        padding: 16px 12px;
        font-size: 13px;
    }
    
    .comparison-table td {
        padding: 16px 12px;
        font-size: 14px;
    }
    
    .comparison-table td:first-child {
        font-size: 15px;
    }
}


.highlight {
    background: #E8F5E8;
    color: #2E7D32;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.warning-highlight {
    background: #fff3cd;
    color: #856404;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.calculation {
    background: #f5f5f5;
    border: 1px solid #4CAF50,;
    border-radius: 4px auto 4px 0;
    padding: 12px;
    font-family: 'Courier New', monospace;
    font-size: 14px;
    color: #333;
    margin: 10px 0;
}

.memory-constraint {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
}
</style>


## Tiling Strategy Options
Several tiling strategies can be used to partition the 256×256 matrices into smaller sub-matrices. The tile size significantly impacts both memory usage and thread utilization, so it must be chosen carefully to achieve optimal performance. from the below listed we can't proceed with some of the options due to limitations we discussed earlier
like number of threads per block , ex:- 64 *64 tile needs 4096 threads per block , but the maximum number of threads per block configuration is 1024, hence we can eliminate the Tile sizes 128*128 & 64*64, for our example we will consider tile size of 32*32.

<div class="tiling-options">
  <div class="tile-card">
    <div class="tile-size">32×32 Tiles</div>
    <div class="tile-count">64 Tiles</div>
    <div class="memory-usage">4 KB per tile</div>
  </div>
  
  <div class="tile-card">
    <div class="tile-size">16×16 Tiles</div>
    <div class="tile-count">256 Tiles</div>
    <div class="memory-usage">1 KB per tile</div>
  </div>
  
  <div class="tile-card">
    <div class="tile-size">8×8 Tiles</div>
    <div class="tile-count">1,024 Tiles</div>
    <div class="memory-usage">256 B per tile</div>
  </div>
  
  <div class="tile-card">
    <div class="tile-size">4×4 Tiles</div>
    <div class="tile-count">4,096 Tiles</div>
    <div class="memory-usage">64 B per tile</div>
  </div>
</div>

### Calculations

<div class="calculation">

Matrix Size: 256 × 256 = 65,536 elements <br>
For tile size T×T: Number of tiles = (256/T)²

• 128×128 tiles: (256/128)² = 2²  = 4 tiles <br>
• 64×64 tiles: (256/64)² = 4²  = 16 tiles <br>
• 32×32 tiles: (256/32)² = 8²  = 64 tiles <br>
• 16×16 tiles: (256/16)² = 16² = 256 tiles <br> 
• 8×8 tiles:   (256/8)²  = 32² = 1,024 tiles <br>
• 4×4 tiles:   (256/4)²  = 64² = 4,096 tiles <br>
</div>




<table class="comparison-table">
<thead>
<tr>
<th>Tile Size</th>
<th>Elements per Tile</th>
<th>Memory per Tile</th>
<th>Threads Needed</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>128×128</strong></td>
<td>16,384</td>
<td>64 KB</td>
<td>16,384</td>
</tr>
<tr>
<td><strong>64×64</strong></td>
<td>4096</td>
<td>16 KB</td>
<td>4096</td>
</tr>
<tr>
<td><strong>32×32</strong></td>
<td>1,024</td>
<td>4 KB</td>
<td>1,024</td>
</tr>
<tr>
<td><strong>16×16</strong></td>
<td>256</td>
<td>1 KB</td>
<td>256</td>
</tr>
<tr>
<td><strong>8×8</strong></td>
<td>64</td>
<td>256 B</td>
<td>64</td>
</tr>
<tr>
<td><strong>4×4</strong></td>
<td>16</td>
<td>64 B</td>
<td>16</td>
</tr>
</tbody>
</table>

# Tiling Step by Step:


<style>
/* Custom CSS for Tiling Step by Step Documentation */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

.tiling-container {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin: 20px 0;
}

.section-header {
    background: linear-gradient(135deg, #4CAF50, #66BB6A);
    color: white;
    padding: 20px 30px;
    font-weight: 600;
    font-size: 20px;
    margin: 0;
}

.section-content {
    padding: 30px;
    background: #fafafa;
}

.intro-text {
    background: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 20px;
    margin: 20px 0;
    border-radius: 4px;
    font-size: 16px;
    color: #555;
}

.configuration-list {
    background: white;
    border-radius: 8px;
    padding: 0;
    margin: 20px 0;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.config-item {
    padding: 18px 25px;
    border-bottom: 1px solid #e8f5e8;
    display: flex;
    align-items: flex-start;
    gap: 15px;
    transition: background 0.2s ease;
}

.config-item:hover {
    background: #f8fffe;
}

.config-item:last-child {
    border-bottom: none;
}

.config-number {
    background: #4CAF50;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}

.config-content {
    flex: 1;
}

.config-title {
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 5px;
    font-size: 16px;
}

.config-description {
    color: #555;
    font-size: 14px;
    line-height: 1.5;
}

.highlight {
    background: #E8F5E8;
    color: #2E7D32;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.calculation {
    background: #f0f0f0;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #333;
    margin: 8px 0;
    display: inline-block;
}

.memory-highlight {
    background: #E3F2FD;
    color: #1976D2;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.dependency-highlight {
    background: #FFF3E0;
    color: #F57C00;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}

.summary-box {
    background: linear-gradient(135deg, #E8F5E8, #F1F8E9);
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 20px;
    margin: 25px 0;
    text-align: center;
}

.summary-title {
    color: #2E7D32;
    font-weight: 600;
    margin-bottom: 10px;
    font-size: 18px;
}

.summary-text {
    color: #388E3C;
    font-size: 14px;
}

.matrix-size-highlight {
    background: #F3E5F5;
    color: #7B1FA2;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
}
</style>


<div class="intro-text">
For our example of <span class="matrix-size-highlight">256×256 matrix</span>, we are splitting the large matrix into tiles (sub-matrices) of size <span class="matrix-size-highlight">32×32</span>, this leads to the following configuration:
</div>

<div class="configuration-list">

<div class="config-item">
<div class="config-number">1</div>
<div class="config-content">
<div class="config-title">Total tiles</div>
<div class="config-description"><span class="highlight">64 tiles</span> (8×8 grid), with each tile calculated by one thread block</div>
</div>
</div>

<div class="config-item">
<div class="config-number">2</div>
<div class="config-content">
<div class="config-title">Tile size</div>
<div class="config-description">Each thread block computes one <span class="highlight">32×32 tile</span>, totaling <span class="calculation">1,024 elements</span></div>
</div>
</div>

<div class="config-item">
<div class="config-number">3</div>
<div class="config-content">
<div class="config-title">Thread allocation</div>
<div class="config-description">Each thread block uses <span class="highlight">1,024 threads</span> (one thread per tile element)</div>
</div>
</div>

<div class="config-item">
<div class="config-number">4</div>
<div class="config-content">
<div class="config-title">Warp organization</div>
<div class="config-description">Each thread block contains <span class="highlight">32 warps</span> <div class="calculation">1,024 ÷ 32 = 32 warps</div></div>
</div>
</div>

<div class="config-item">
<div class="config-number">5</div>
<div class="config-content">
<div class="config-title">Memory footprint</div>
<div class="config-description">Each thread block loads <span class="memory-highlight">32×32 elements = 4 KB</span> per matrix (float32 precision)</div>
</div>
</div>

<div class="config-item">
<div class="config-number">6</div>
<div class="config-content">
<div class="config-title">Dependencies</div>
<div class="config-description">To calculate one tile <span class="dependency-highlight">C(0,0)</span>, it requires <span class="dependency-highlight">8 tiles from matrix A</span> (row 0) and <span class="dependency-highlight">8 tiles from matrix B</span> (column 0)</div>
</div>
</div>

<div class="config-item">
<div class="config-number">7</div>
<div class="config-content">
<div class="config-title">Loading strategy</div>
<div class="config-description">Tiles from A and B are <span class="highlight">cooperatively loaded by warps</span> into the thread block's shared memory</div>
</div>
</div>

<div class="config-item">
<div class="config-number">8</div>
<div class="config-content">
<div class="config-title">Memory sharing</div>
<div class="config-description">Data in <span class="memory-highlight">shared memory</span> is accessible to all warps within the thread block</div>
</div>
</div>

<div class="config-item">
<div class="config-number">9</div>
<div class="config-content">
<div class="config-title">Sequential processing</div>
<div class="config-description">Tiles of A and B are loaded in <span class="highlight">8 sequential phases</span> as described below</div>
</div>
</div>

</div>

<div class="summary-box">
<div class="summary-title">🎯 Key Observation</div>
<div class="summary-text">The tiling approach enables efficient use of shared memory by loading small, manageable chunks of data that fit within memory constraints while maximizing thread utilization and computational efficiency.</div>
</div>

---



> **Next Steps**: We'll examine how Thread Block (0,0) processes these tiles through the 8 sequential phases to compute C(0,0) using shared memory optimization.

> Full Grid Execution: This visualization represents the computation of a single tile C(0,0). Simultaneously, the remaining 63 thread blocks execute identical processes to compute their assigned tiles, covering the complete 8×8 tile grid from C(0,0) to C(7,7). The collective output of all 64 thread blocks yields the final 256×256 matrix containing 65,536 elements.


![tiled_matrix_multiplication](/images/cuda/tile_gemm/tiled_matrix_multiplication.png)


![tile_c00](/images/cuda/tile_gemm/tile-c00-generation.png)

![phase1_tile_c00](/images/cuda/tile_gemm/phase-1-complete-tile-view.png)

![phase2_tile_c00](/images/cuda/tile_gemm/phase-2-complete-tile-view.png)

![phase3_tile_c00](/images/cuda/tile_gemm/phase-3-complete-tile-view.png)

![phase4_tile_c00](/images/cuda/tile_gemm/phase-4-complete-tile-view.png)

![phase5_tile_c00](/images/cuda/tile_gemm/phase-5-complete-tile-view.png)

![phase6_tile_c00](/images/cuda/tile_gemm/phase-6-complete-tile-view.png)

![phase7_tile_c00](/images/cuda/tile_gemm/phase-7-complete-tile-view.png)

![phase8_tile_c00](/images/cuda/tile_gemm/phase-8-complete-tile-view.png)




# Data Reuse Analysis:  Naive GEMM vs Tile GEMM

Let us examine how tiling solves the data reuse inefficiencies that arise in naive GEMM cache evictions.

Configuration Recap:

**Naive GEMM:** Grid of 1,024 thread blocks (32×32 grid), each thread block containing 1,024 threads (32×32), with no shared memory usage.

**Tiled GEMM:** Grid of 64 thread blocks (8×8 grid), each thread block containing 1,024 threads (32×32), using 32×32 tiles with shared memory for matrices A and B.

If we revisit the configuration and execution for Naive GEMM , our configuration comprised  Grid of 64 Thread Blocks in 2D (32,32) &  Thread Block comprised of 1024 threads in 2D (32,32) , each Thread block had 8 Warps, each warp calculating 32 elements in output matrix.

## Data Reuse Analysis

**Naive GEMM Limitations:**

To calculate adjacent output elements C(0,0) and C(0,1), both require the entire first row of matrix A. While this row is loaded once from global memory for C(0,0), cache evictions prevent reuse for C(0,1), forcing redundant global memory accesses.

In order to calculate C(0,0) and C(1,0) we need the first column of B matrix i.e B(0,0) => B(0,31), while calculating C(0,0) this whole column had been fetched from Global memory and used for C(0,0) the same could not be reused for performing computations for  C(1,0) due to cache evictions, we will briefly look at the steps in Warp-0

### Memory Access per Thread Block (Naive):

Load Row-0 from Matrix A: 256 × 4 bytes = 1 KB
Load Columns 0-31 from Matrix B: 256 × 32 × 4 bytes = 32 KB

#### Data Loading from Global Memory:
- Matrix A: 1 row = 256 elements × 4 bytes = 1,024 bytes
- Matrix B: 1 column = 256 elements × 4 bytes = 1,024 bytes
- Total: 2,048 bytes

#### Operations Performed:

256 multiply operations + 256 add operations = 512 FLOPs

    Arithmetic Intensity:
    512 FLOPs ÷ 2,048 bytes = 0.25 FLOPS/Byte

**Problem: Matrix B columns cannot be reused across computations due to cache evictions**

### Tiled GEMM Solution:

#### Data Loading from Global Memory:
- Load 32×32 tile from Matrix A into shared memory: 32 × 32 × 4 bytes * 8 Tiles  = 32 KB
- Load 32×32 tile from Matrix B into shared memory: 32 × 32 × 4 bytes * 8 Tiles  = 32 KB
- Total: 64 KB = 65,536 bytes

#### Operations Performed:

32×32 threads × 512 operations each = 524,288 FLOPs

    Arithmetic Intensity:
    524,288FLOPs ÷ 65,536 bytes = 8.0 FLOPS/Byte

**Key Observation:**
The improvement comes from data reuse within shared memory. In tiled GEMM, each byte loaded from global memory is reused multiple times across different computations, while in naive GEMM, each byte is used only once before being potentially evicted from cache.



![naiv_gemm_cache_evictions](/images/cuda/tile_gemm/naive_gemm_cache_eviction_issues.png)

![tile_gemm_reuse](/images/cuda/tile_gemm/tile_gemm_shared_memory_reuse.png)

# Performance Analysis 

To understand optimizations achieved by Tiled GEMM, we need to understand the theoritical limits of Hardware to calculate Arthimatic Inensity for both
Naive & Tiled GEMM

## A100 Specifications

- CUDA Cores: 6,912 (108 SMs × 64 cores per SM)
- Base Clock: ~1.41 GHz
- Memory: 80GB HBM2e
- Memory Interface: 5,120-bit bus width
- Memory Clock: ~1.6 GHz (effective)

## Peak FLOPS Calculation

- Cores per SM: 64 CUDA cores
- Total SMs: 108 streaming multiprocessors
- Total CUDA cores: 108 × 64 = 6,912 cores
- Base clock frequency: ~1.41 GHz
- Operations per core per clock: 1 FMA = 2 FLOPs

## Peak FP32 Performance:

  Peak FLOPS = Total Cores × Clock Frequency × FLOPs per Clock <br>
  Peak FLOPS = 6,912 × 1.41 × 10⁹ × 2 <br>
  Peak FLOPS ≈ 19.5 TFLOPS <br>

## Peak Memory Bandwidth Calculation  

  Memory interface width: 5,120 bits = 640 bytes <br>
  Memory clock (effective): ~1,600 MHz (DDR, so 800 MHz × 2)

## Peak Memory Bandwidth:

 Peak Bandwidth = Interface Width × Memory Clock</br>
 Peak Bandwidth = 640 bytes × 1,600 × 10⁶ transfers/second</br>
 Peak Bandwidth ≈ 2,039 GB/s ≈ 2.0 TB/s

## Peak Arithmetic Intensity

Arithmetic Intensity =  FLOPS ÷  Memory Bandwidth <br>
                       = 19.5 * 10^12 FLOPS / 2.0 * 10^12 Bytes per sec <br>
                       = 19.5/2 = 9.75 Flops/Byte 

## Arithemetic Intensity  :

To calculate each cell in the output matrix we need to fetch 256 elements from A & 256 Elements from B , and perform 256 Multiply and 256 Additions


<div style="
  border: 2px solid #4CAF50; 
  border-radius: 8px; 
  overflow: hidden; 
  margin: 1em 0;
">

  <div style="
    background-color: #4CAF50; 
    color: white; 
    font-weight: bold; 
    text-align: center; 
    padding: 6px 0;
  ">
    Naive GEMM
  </div>

  <!-- White code block -->
  <pre style="
    background-color: #ffffff; 
    color: #2d2d2d; 
    padding: 16px; 
    margin: 0; 
    font-size: 0.95rem; 
    line-height: 1.5; 
    overflow-x: auto;
  ">
<b>C(0,0) = A(0,0) * B(0,0) + A(0,1) * B(1,0) + A(0,2) * B(2,0) + ... + A(0,256) * B(256,0)</b>

<b>FLOPS = 256 Multiply  + 256 Additions = 512 FLOPS</b>
<b>Bytes Transferred = 256 * 4 Bytes (A) + 256 * 4 Byes (B) = 2 KB = 2048 Bytes</b>

<b>FLOPS/Bytes = 512 / 2048 = 0.25 FLOPS/Byte</b>
  </pre>
</div>



To calculate each cell in the output matrix we need to load 8 Tiles each from A & B over several phases, each phase includes 32 Multiply & 32 Addition operations.



<div style="
  border: 2px solid #4CAF50; 
  border-radius: 8px; 
  overflow: hidden; 
  margin: 1em 0;
">

  <div style="
    background-color: #4CAF50; 
    color: white; 
    font-weight: bold; 
    text-align: center; 
    padding: 6px 0;
  ">
    Tiled GEMM
  </div>

  <!-- White code block -->
  <pre style="
    background-color: #ffffff; 
    color: #2d2d2d; 
    padding: 16px; 
    margin: 0; 
    font-size: 0.95rem; 
    line-height: 1.5; 
    overflow-x: auto;
  ">

<b>Per Phase (32×32 tile):</b><br>
<b>FLOPs = 32 multiply + 32 add = 64 FLOPs per thread </b>
<b>Total per phase = 1,024 threads × 64 FLOPs = 65,536 FLOPs</b>

<b>Total (8 phases):</b> <br>
<b>Total FLOPs = 65,536 × 8 = 524,288 FLOPs </b>
<b>Data loaded = 64 KB (32 KB A + 32 KB B across all phases)</b>
<b>Arithmetic Intensity = 524,288 ÷ 65,536 = 8.0 FLOP/byte</b>
   </pre>
</div>

### Performance Summary

| Approach | Arithmetic Intensity | Memory Efficiency | Category |
|----------|---------------------|-------------------|----------|
| **Naive GEMM** | `0.25 FLOP/byte` | `2.5%` | Memory-bound |
| **Tiled GEMM** | `8.0 FLOP/byte` | `82%` | Near compute-bound |

**Key Improvement:** 32× arithmetic intensity gain transforms kernel from memory-bound towards compute-bound operation.

