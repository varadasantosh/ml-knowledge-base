---
title: "Flash Attention"
---


<!--more-->


# Problems with Naive Attention

Flash Attention is IO Aware & Exact Attention. To understand this, we need to be aware of Vanilla Attention (Self-Attention), which is pivotal for Transformer Architecture. Additionally, having some knowledge of GPU Architecture is beneficial.

**Self-Attention Recap**: In order to calculate Self-Attention, the following steps are performed:

  
   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
   $$

   1. The input embeddings `x` with dimensions (batch_size, sequence_len, n_dim) are passed through three linear layers with weights  $W_q$ ,  $W_k$  &  $W_v$ . As a result, we obtain the matrices **Q**, **K**, and **V**, which have the same dimensions as `x`:

      - \( Q \): Query matrix
      - \( K \): Key matrix
      - \( V \): Value matrix
      - \( d_k \): Dimensionality of the key vectors

 
   2. **Q** - Query Matrix &  **K** -Key Matrix are moved to SM (Streaming Multiprocessor) On-chip Memory for Matrix Multiplication Operation (`GEMM`), Result of this operation is moved to HBM(High Bandwidth Memory) in GPU
   
   3.  We need to apply Masking on the result of Multiplication of **Q** & **$${K^T}$$** to ensure padding tokens get zero probabilities after applying softmax ,
       this result again needs to be moved from HBM to SM On-Chip Memory.
   4.  After applying Masking operation, the same matrix is moved from On-chip Memory to HBM
   5.  Next step would be to apply Softmax operation on the matrix whose size is (batch_size,seq_len,seq_len), to apply softmax the matrix is moved from HBM to On-chip memory.    
   6.  After the Softmax is calculated , result of the same is moved to HBM(High Bandwidth Memory), The size of the Softmax matrix would be of **(batch_size,seq_len,seq_len)**
   7.  Next step is to perform Matrix multiplication between the probabilities(Normalizing the dot product between Q,K) calculated in earlier step using Softmax & the **V** Values matrix whose size is **(batch_size,seq_len,n_dim)**, hence these both matrices need to be moved from HBM to On-Chip memory
   8.  Matrix multiplication is performed between Softmax Values & **V** values matrix to get the final attention score

   From the above steps we can infer that majorly the there are two types of operations one being Matrix Multiplications which is FLOPS(Floating Point Operations), other is data movement       
   between DRAM(HBM) to SRAM (On-Chip Memory), due to massive parallel processing capabilities of GPU Floating point operations are calculated faster , once this is done threads present inside 
   the  **SM** are idle until they get new set of instructions and Data on which these instructions need to be performed , **this makes these operations Memory bound as the time taken to move 
   the data between SRAM (On Chip Memory) & DRAM  is more than the time taken to perform FLOPS (Matrix Multiplicaton in this case)**
   
# Solution - Flash Attention

   Flash Attention address this problem by dividing the matrices into multiple blocks , and peforms fusing of kernal operations ( Kernels are functions) , fusing Kernel operations can be 
   considered as chaining different functions on each set of blocks, this fusing of kernel operation reduces the need for storing of intermediate results and memory transfers, also the same 
   calculations are recomputed during backward propagation , instead of storing them and moving them between memory layers, though these two operations increase the number of FLOPS the time 
   taken to calculate the attention matrix is less duration this reduces the I/O operations which is bottleneck in Self 
   Attention.

   
   Flash attention divides the matrix into small tiles and the operations like dot product between Q,${K^T}$ are performed and result of this is passed to another kernel function which 
   calculates mask & passes the output to another function that calculates softmax , furhter this result is passed to another kernel which calculates the dot product between softmax values and 
   V matrix, as these data is passed through multiple kernel functions within SRAM we don't store the intermediate results on HBM.

   
   But here lies the major challenge, inorder to calculate the Softmax we need all the values at once to perfrom sum operation  which is required to calculate(denominator), this is required 
   as we need to divide each element of the dot matrix by sum of all the elments(which is Softmax formula) , as we are dividing the matrix into multiple blocks to perfrom kernel fusion 
   (chaining kernel functions like Dot product, masking and Softmax ) calculating the total sum is not possible ,  hence we need a way to calculate the softmax for these batches accurately,
   fortunately this can be addressed calculatin online softmax, which uses tiling technique which is metioned in NVIDIA researcher [paper](https://arxiv.org/abs/1805.02867), this approach 
   allow us to calculate the softmax for individual blocks and when we are merging them we incrementally calculate the final softmax using the formaula mentioned below until we reach final 
   merging on all the blocks 
   
   <img width="492" alt="image" src="https://github.com/user-attachments/assets/c5f58700-db05-4027-ab1b-a0814ec769cc" />


   $$
   \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}
   $$
   
  **Few intresting points to note here is the number of FLOPS (Floating point operations) are more in number than the Self Attention , but the time taken is less compared to Self Attention as we are working on small cunks which makes it faster move the data between HBM and On-Chip memory and On-chip Memory to HBM memory , as we are dividing into multiple chunks this also allows us to increase Sequence Lenght which is Context Length of model, hence we can have more context length for the training model**. 

  # [Online Softmax Calculation](https://github.com/varadasantosh/deep-learning-notes/blob/tensorflow/Flash_Attention_Calculations(Online_Softmax).ipynb) 

  [Notebook](https://colab.research.google.com/github/varadasantosh/deep-learning-notes/blob/tensorflow/Flash_Attention_Calculations(Online_Softmax).ipynb)
   
   Reference Links:-

   1. https://horace.io/brrr_intro.html
   2. https://training.continuumlabs.ai/inference/why-is-inference-important/flash-attention-2
   3. https://www.youtube.com/watch?v=IoMSGuiwV3g
   4. https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad#:~:text=So%20basically%2C%20in%20order%20to,statistics%20for%20each%20of%20the
   5. https://www.nvidia.com/en-us/on-demand/session/gtc24-s62546/


# Illustration of Standard Attention vs Flash Attention from Hugging Face:-
   ------------------
   ![image](https://github.com/user-attachments/assets/8ce6ec2f-2df2-4d5e-b643-598ba3b27097)
