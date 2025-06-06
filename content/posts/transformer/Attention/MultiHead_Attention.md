
---
title: "Peek Into MultiHead Attention"
---


<!--more-->

# MultiHead Attention

The Multi Head Attention is an extension to Self Attention, while the Self Attentin defined in Transfomers helps us to overcome limitations faced by RNN's, if we look at the above pitcutre we are calculating the attention over all heads of Llama Model , Multi Head Attention helps us to attend diferent aspects of elements in a sequence, in such case single weighted average is not
good option, to understand different aspects we divide the Query, Key & Value matrices to different Heads and calculate the attention scores of each head, to calculate attention for each head
we apply the same approach mentioned above,after the attention scores of each head are calculated we concatenate the attention scores of all the heads, this approach yeilds better results than
finding the attention as a whole, during this process the weight matrices that are split are learned for each head.

Llama Model(`Llama-3.2-3B-Instruct`) referred above has Embedding Dimensions of size - **3072** &  Number of Heads - **24**, thus our Query , Key & Values are split into 24 heads each head would be of size 3072/24 = 128

Multihead(Q,K,V) =  Concat($head_{1}$, $head_{2}$,.....$head_{24}$)
$head_{i}$ = Attention(Q$W_{i}$^Q, K$W_{i}$^K, V$W_{i}$^V)

Below image illustrates the process of calculating MultiHead Attention

![image](https://github.com/user-attachments/assets/67f3bb0b-1ac6-454e-ab5c-6975c368a9e3)

# Reference Implementation


```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model,num_heads):

        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q= nn.Linear(d_model,d_model)
        self.W_k= nn.Linear(d_model,d_model)
        self.W_v= nn.Linear(d_model,d_model)
        self.W_o= nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self,Q,K,V,mask=None):

        attention_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0,-1e9)

        attention_probs = torch.softmax(attention_scores,dim=-1)
        output = torch.matmul(attention_probs,V)
        return output

    def split_heads(self,x):
        batch_size,seq_length,d_model = x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)

    def combine_heads(self,x):
        batch_size,_,seq_length,d_k = x.size()
        x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)

    def forward(self,Q,K,V,mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q,K,V,mask)
        output = self.W_o(self.combine_heads(attention_output))
        return output

```

# Visulization of MultiHead Attention using Llama Model

We use the same Model & Input text we considered for Self Attention and look at one of the Heads of the Last layer in to see if they are able to attend contextual relation ships between different tokens

## Download the Llama Model from Hugging Face


```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

model_name= "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
```


    tokenizer_config.json:   0%|          | 0.00/54.5k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/9.09M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/296 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/878 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/20.9k [00:00<?, ?B/s]



    Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]



    model-00002-of-00002.safetensors:   0%|          | 0.00/1.46G [00:00<?, ?B/s]



    model-00001-of-00002.safetensors:   0%|          | 0.00/4.97G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


Llama Model Architecture , Llama model being Decoder only Transformer we can see there are only Decoder Layers (28 Layers), zero encoder layers


```python
print(model)
```

    LlamaModel(
      (embed_tokens): Embedding(128256, 3072)
      (layers): ModuleList(
        (0-27): 28 x LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
            (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
            (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
            (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
            (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
            (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
          (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        )
      )
      (norm): LlamaRMSNorm((3072,), eps=1e-05)
      (rotary_emb): LlamaRotaryEmbedding()
    )


## Tokenize the Input Sentence & Pass it through the Llama Model


```python
import torch

text = "the financial bank is located on river bank"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
token_ids = inputs.input_ids[0]
tokens = tokenizer.convert_ids_to_tokens(token_ids)
model = model.to("cuda")
with torch.no_grad():
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
```

    `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.


Get The Attention Matrix from the Outputs, we can see the below dimensions of the `attention_matrix` of length 28 and infer that there are 28 attention layers which we can also see from model architecture & each layer's attention matrix is of shape (1,24,9,9) - This is because Llama Model has 24 Heads (This refers to Multi Head attention) and sequence length
of tokens that we passed is of length 9 hence the dimension of each head is 9*9


```python
attention_matrix = outputs.attentions
print(f" Number of Attention Matrices == Number of Layers{len(attention_matrix)}")
print(f" Shape of Each Attention Matrix {attention_matrix[0].shape}")
```

     Number of Attention Matrices == Number of Layers28
     Shape of Each Attention Matrix torch.Size([1, 24, 9, 9])


Get the last Layer from the Attention matrix and draw a heatmap for each head, as the Llama model we are referring has 24 heads below code snippet , will produce 24 heatmaps for each heads but due to lack of space and to keep it simple we will look at one of the heads , which captures the context of the same token `bank` according to its position and context


```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(24, 1, figsize=(10, 200))
for i, ax in enumerate(axes.flat):
    sns.heatmap(attention_matrix[27][0][i].cpu(), ax=ax, cmap="viridis",annot=True,fmt=".2f",xticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),yticklabels=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) )
    ax.set_title(f"Head {i+1}",fontdict={'fontsize':25})
plt.tight_layout()
plt.show()
```


    
![png](/images/transformers/Attentions/multihead_attention.png)
    


# Focus on One of Attention Head

![image](https://github.com/user-attachments/assets/a08027b7-2a41-4578-b437-2f4f0d84a65a)

![image](https://github.com/user-attachments/assets/d3eb0e3d-7e84-420f-a070-92eb634b45ff)
