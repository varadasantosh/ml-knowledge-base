

---
title: "Distributed Training"
---


<!--more-->

Before diving deeper into understanding different techniques of Distributed Training, it is essential to understand why it is needed.

With advancements in both technology and hardware & availability of data  the size of deep learning models has grown significantly. Modern Large Language Models (LLMs) are trained on massive datasets and have billions of parameters, making them too large to fit within the memory of a single GPU.

If such models were trained on a single GPU, the process could take hundreds of years to complete. Distributed training techniques, such as Fully Sharded Data Parallel (FSDP), help overcome these limitations by distributing the workload across multiple GPUs. This not only accelerates training but also enables the development of increasingly larger and more capable models. The more data a model can learn from, the better its performance.

A model comprises of below 
- Parameters (Weights) - Calculated during Forward Propagation
- Gradients - Calculate during Backward Propagation
- Optimizer State (Ex:- Adam Optimizer has additionally has 3 more parameters Momentum, Velocity )
- Token Embeddings
- Positional Embeddings
  

NCCL Primitives - [DOCS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
---------------
- Broadcast
- Reduce
- All Reduce
- All Gather
- Reduce Scatter

Distributed Data Parallel 
----------------------------
Deep Learning models consist of two main components mentioned below . Distributed Data Parallel (DDP) helps improve training speed, particularly when the number of parameters is relatively small, but the dataset is large.
- parameters (model weights)
- data. 

When a dataset is too large to fit into GPU VRAM, there are two main options:

- Scaling the infrastructure (adding more GPUs or nodes). However, this has limitations since GPU VRAM cannot be scaled indefinitely.
- Dividing the dataset into smaller batches so that each batch fits into the available VRAM.

While batching allows training on large datasets, training sequentially (one batch at a time) can be inefficient and slow. This is where Distributed Data Parallel (DDP) comes into play.

With DDP, instead of processing batches sequentially, we distribute batches across multiple GPUs and train in parallel. For example, if we have 4 GPUs in a single node, we can divide the dataset into 4 batches and assign one batch to each GPU.

To enable this, we need to replicate the model across all GPUs, ensuring each GPU has an identical copy of the model. Each GPU processes a different batch of data independently. After processing, gradients are synchronized across all GPUs using an all-reduce operation (**NCCL Library** ), ensuring model updates remain consistent. The same can also be extended to GPU across different Nodes.

There is lot happening behind the scenes for co-ordinating the training process between GPU's (Intra Node) & Inter Node. Below are the high- level steps that are performed

1. Divide the Batches across GPU's
2. Go through the Forward Pass (Each Batch that resides on respective GPU)
3. Calculate Local Gradients (on Each GPU)
4. Perform All Reduce Operation to bring all the Local Gradients to One of the GPU s'
5. Once the Gradients are accumulated and calculated , pass the Gradients back to all the GPU's
6. Each GPU calculates peforms the Update Optimizater State for the corresponding Weights (Parameters)

<img width="336" alt="image" src="https://github.com/user-attachments/assets/dd1c3bf1-f629-49c7-8d5c-ca92707426aa" />


Pipeline Parallelism
---------------------------
As we briefly looked at the Distributed Data Parallel, it address the challenges with Model Training with Large Dataset and Model can fit on single GPU, after the birth of Transformer Architecture we evidenced unprecednted increase in the size of the model , each Model has large number
of parameters, if the model can't be fit into memory **Distributed Data Parallel** alone would not solve the problem as this approach relies on fitting entire model in Memory, Pipeline Parllelsim to the rescue which pivot the model to be ditributed across GPU's rather than distributing data, Pipeline Parallelism as a concept can be further implemented in two different ways

 - Vertical splitting the Model (**Model Parallelism**)

    In this approach the layers of the model are split across the available GPU's ex:- if we have 4 Hidden Layers and 4 GPU's split layers and train each layer on one GPU,
    the downside of this approach is while one layer is being trained on one GPU the other GPU sits idle , which is not efficient way of using the precious resources.
   
 - Horizontal Spllitting of Model(**Tensor Parallelism**)

    Here the parameters, Gradients & Optimizer States are split across multiple GPU's meaning if we need to calcualte Dot Product W.X, We split these matrices across 4 GPU's and calculate
    the dot product across different GPU's and bring the Parameters to one GPU for calculating the Gradients for Backward Propagation using NCCL Operations


# Steps performed during FSDP

Let us consider the Model Architure with below configuration
```
. Number of GPU's- 4
. Number of Layers - 2
. Number of Neurons in each Hidden Layer-4 
. Input Features - 4
. Total Number of rows input data - 12
. 12 rows divide across 4 GPU - Each GPU gets batch of 3 rows
```
Two Layers , each with Matrices $W_{1}$, $W_{2}$ with below dimensions
- $W_{1}$ -  4 * 4 (Input Features- 4, Number of Neurons in Each Layer-4)
- $W_{2}$ -  4 * 4 (Output from Activation of First Layer, Number of Neurons in Each Layer-4)


# Layer 1 - Forward Propagation:
- Input Data on Each GPU , each GPU has data of Batch size of 3 Rows & 4 Columns(features) 3 * 4 matrix
- $w_{1}$ is split across 4 GPU's , each GPU has weight matrix of size 1 * 4
- Peform **All Gather** Collective Operation to get all the Weights of the corresponding Shard & Layer
- After **All Gather** Operation all GPU's has full weight matrix of Layer-1 to proceed with **GEMM** like mentioned below
- GPU-0 - $\hat{y_{0}}$ = $w_{1}$ * $x_{0}$ + $b_{1}$
- GPU-1 - $\hat{y_{1}}$ = $w_{1}$ * $x_{1}$ + $b_{1}$
- GPU-2 - $\hat{y_{2}}$ = $w_{1}$ * $x_{2}$ + $b_{2}$
- GPU-3 - $\hat{y_{3}}$  = $w_{1}$ * $x_{3}$ + $b_{3}$
- After calculating **GEMM** and **Activations**, Weights gathered from other GPU's for the corresponding shard are freed, each GPU would remain
  with weight matrix of 1 * 4 size
- Activations are stored in each GPU, these are required while calculating Gradients during Backward Propagation

# Layer 2 - Forward Propagation: 
- Steps would remain same like in Layer-1
- Output of Layer-1 would be passed as input to Layer-1
- Output of Layer-1 is of Size 3 * 4 ($x_{0}$ =  3 * 4  & $w_{1}$ =  4 * 4 )
- Layer2 has 4 features as input and it has 4 hidden Layers hence the weight matrix $w_{2}$ is of shape 4 * 4
- Like in Layer-1 $w_{2}$ is split across 4 GPU's , each GPU has weight matrix of size 1 * 4
-  Peform **All Gather** Collective Operation to get all the Weights of the corresponding Shard & Layer
-  After **All Gather** Operation all GPU's has full weight matrix of Layer-2 to proceed with **GEMM** like mentioned below
-  After calculating **GEMM** and **Activations**, Weights gathered from other GPU's for the corresponding shard are freed,
   each GPU would remain with weight matrix of 1 * 4 size
- Activations are stored in each GPU, these are required while calculating Gradients during Backward Propagation

# Layer2 - Backward Propagation:
- Peform **All Gather** Operation on Layer-2 for gathering all weights of the Shard & Layer, this is required for Gradient Calculation
- After **All Gather** Weights of Layer-2 are present on all the GPU's allowing us to perform Gradient Calculations
- Each GPU perform Gradient Calculation Locally
- But the Gradients calculated on each GPU are partial,as each of them are working on different batch of data, hence the gradient needs to be
  aggregated
- to achieve the aggregation of gradients from all GPU's and sending the relevant gradients for each GPU to be adjust we take help of
  **Reduce Scatter** operation (Refer to NCCL Operations)
- After the **Reduce Scatter** Operation each GPU now have the Gradients for Layer-2 each Gradient matrix of size  1 * 4


# Layer1 - Backward Propagation:
- Backward Propagation Remains as above 
- Peform **All Gather** Operation on Layer-1 for gathering all weights of the Shard & Layer, this is required for Gradient Calculation
- After **All Gather** Weights of Layer-1 are present on all the GPU's allowing us to perform Gradient Calculations
- Each GPU perform Gradient Calculation Locally
- But the Gradients calculated on each GPU are partial,as each of them are working on different batch of data, hence the gradient needs to be
  aggregated
- to achieve the aggregation of gradients from all GPU's and sending the relevant gradients for each GPU to be adjust we take help of
  **Reduce Scatter** operation (Refer to NCCL Operations)
- After the **Reduce Scatter** Operation each GPU now have the Gradients for Layer-1 each Gradient matrix of size  1 * 4
- This gradients are now would be used to perform Optimizer Update to adjust the weights

**Important Note**    
 During the Process the Weight Updates are performed after Complete Forward Pass & Backward Pass, Though the Gradeints for
 Layer-2 are calculated Before Layer-1 Gradeints, we can't update Weights for Layer-1 until the Gradeints for Layer-1 are calculated
 because if we update the Weights for Layer-2 , it would impact the Caclulation of Gradients for Layer-1 hence the weights are updated
 only after completed Backward Pass through from last layer to first layer, after this cycle the Weights can be updated across all the 
 layers
  
  
# FSDP Workflow:-
  ---------------
  ![image](https://github.com/user-attachments/assets/3b773598-b5b8-457c-aa45-c6995935c641)

# Reference Articles
  - https://tinkerd.net/blog/machine-learning/distributed-training/
  - https://www.youtube.com/watch?v=toUSzwR0EV8&t=2s
  - https://github.com/huggingface/blog/blob/main/pytorch-fsdp.md
  - https://blog.clika.io/fsdp-1/



