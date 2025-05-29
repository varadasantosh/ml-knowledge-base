
---
title: "Dataset Preparation"
---


<!--more-->

In this section, we will cover:
1. Tokenization
2. Data Augmentation
3. Dataset Formatting


<a href="https://colab.research.google.com/github/varadasantosh/deep-learning-notes/blob/tensorflow/transformers/prepare_dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
!pip install datasets
!pip install huggingface_hub
```

    Requirement already satisfied: datasets in /usr/local/lib/python3.11/dist-packages (3.4.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from datasets) (3.18.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from datasets) (2.0.2)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (18.1.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from datasets) (2.2.2)
    Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.11/dist-packages (from datasets) (2.32.3)
    Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.11/dist-packages (from datasets) (4.67.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.11/dist-packages (from datasets) (3.5.0)
    Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.70.16)
    Requirement already satisfied: fsspec<=2024.12.0,>=2023.1.0 in /usr/local/lib/python3.11/dist-packages (from fsspec[http]<=2024.12.0,>=2023.1.0->datasets) (2024.12.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from datasets) (3.11.14)
    Requirement already satisfied: huggingface-hub>=0.24.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.29.3)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from datasets) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from datasets) (6.0.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (6.2.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (0.3.0)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.18.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24.0->datasets) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.32.2->datasets) (2025.1.31)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.1)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.17.0)
    Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.11/dist-packages (0.29.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (3.18.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (2024.12.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (6.0.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (2.32.3)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (4.67.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub) (2025.1.31)


# Import Libraries


```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
```

# Define Generator for Training Tokenizer


```python
def get_ds(ds,lang):
  for sentence in ds['translation']:
    yield sentence[lang]
```

# Build Tokenizer


```python
def build_tokenizer(ds,lang):
  tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
  tokenizer.pre_tokenizer = Whitespace()
  trainer = WordLevelTrainer(special_tokens=["[UNK]","[SOS]","[EOS]","[PAD]"],min_frequency=2)
  tokenizer.train_from_iterator(get_ds(ds,lang),trainer)
  return tokenizer
```

# Train the Tokenizer on Dataset


```python
raw_ds = load_dataset('opus_books','en-fr',split='train')
src_tokenizer = build_tokenizer(raw_ds,'en')
tgt_tokenizer = build_tokenizer(raw_ds,'fr')
```

# Understand the Tokenizer Behaviour


```python
src_tokenizer.get_vocab_size()
```




    30000




```python
from collections import OrderedDict
vocab = src_tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(),key = lambda x:x[1])
ordered_vocab=OrderedDict(sorted_vocab)
```


```python
example_sentence = raw_ds['translation'][5]['en']
tokens= src_tokenizer.encode(example_sentence).tokens
token_ids = src_tokenizer.encode(example_sentence).ids
id_token_pair = [f"{(token,ordered_vocab[token])}" for token in tokens]
print(f"Tokens:-{tokens}")
print(f"Token ID:-{token_ids}")
print(f"ID-Token Pair:-{id_token_pair}")

```

    Tokens:-['He', 'arrived', 'at', 'our', 'home', 'on', 'a', 'Sunday', 'of', 'November', ',', '189', '-.']
    Token ID:-[66, 642, 29, 113, 454, 30, 10, 2524, 7, 3305, 4, 21600, 16245]
    ID-Token Pair:-["('He', 66)", "('arrived', 642)", "('at', 29)", "('our', 113)", "('home', 454)", "('on', 30)", "('a', 10)", "('Sunday', 2524)", "('of', 7)", "('November', 3305)", "(',', 4)", "('189', 21600)", "('-.', 16245)"]


# Build Dataset for preparing Training Data


```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

  def __init__(self, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, ds, seq_len):

    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.ds = ds
    self.seq_len = seq_len
    self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOC]")],dtype=torch.int64)
    self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")],dtype=torch.int64)
    self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")],dtype=torch.int64)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self,idx):

    sentence = self.ds['translation'][idx]
    src_sentence = sentence['en']
    tgt_sentence = sentence['fr']

    src_tokens = self.src_tokenizer.encode(src_sentence).ids
    tgt_tokens = self.tgt_tokenizer.encode(tgt_sentence).ids

    src_pad_tokens = self.seq_len - len(src_tokens) - 2
    tgt_pad_tokens = self.seq_len - len(tgt_tokens) - 1

    enc_input =  torch.cat([ self.sos_token,
                             torch.tensor(src_tokens,dtype=torch.int64),
                             self.eos_token,
                             torch.tensor([self.pad_token]*src_pad_tokens,dtype=torch.int64)
                             ],dim=0)

    dec_input = torch.cat([self.sos_token,
                           torch.tensor(tgt_tokens,dtype=torch.int64),
                           torch.tensor([self.pad_token]*tgt_pad_tokens,dtype=torch.int64)
                           ],dim=0)

    label = torch.cat([ tgt_pad_tokens,
                       self.eos_token,
                       torch.tensor([self.pad_token]*tgt_pad_tokens,dtype=torch.int64)
                      ],dim=0)

    return {

            "encoder_input" :  enc_input,
            "decoder_input" : dec_input,
            "label" : label,
            "encoder_mask": (enc_input!=self.pad_token).int(),
            "decoder_mask": (dec_input!=self.pad_token).int() & casual_mask(dec_input.size(0)),
            "src_sentence":  src_sentence,
            "target_sentence": tgt_sentence

    }

  def casual_mask(size):
    mask = torch.tiru(torch.ones(1,size,size),diagonal=1,dtype=torch.int64)
    return mask==0

```

# How Masking Works


```python
seq_len = 20
example_src_sentence = raw_ds['translation'][5]['en']
example_tgt_sentence = raw_ds['translation'][5]['fr']
sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")],dtype=torch.int64)
eos_token= torch.tensor([src_tokenizer.token_to_id("[EOS]")],dtype=torch.int64)
pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")],dtype=torch.int64)
enc_input_tokens = src_tokenizer.encode(example_src_sentence).ids
dec_input_tokens = tgt_tokenizer.encode(example_tgt_sentence).ids
num_enc_pad_tokens = seq_len - len(enc_input_tokens) - 2
num_enc_pad_tokens = seq_len - len(dec_input_tokens) - 1

enc_input = torch.cat([sos_token,
           torch.tensor(enc_input_tokens,dtype=torch.int64),
           eos_token,
           torch.tensor([pad_token]* num_enc_pad_tokens,dtype=torch.int64)
           ],dim=0)

dec_input = torch.cat([sos_token,
                       torch.tensor(dec_input_tokens,dtype=torch.int64),
                       torch.tensor([pad_token]* num_enc_pad_tokens,dtype=torch.int64)
                       ],dim=0
                       )

```

# Encoder Masking


```python
enc_input
```




    tensor([    1,    66,   642,    29,   113,   454,    30,    10,  2524,     7,
             3305,     4, 21600, 16245,     2,     3,     3,     3,     3,     3,
                3,     3,     3,     3])




```python
enc_input.size(0)
```




    24




```python
mask = (enc_input!=pad_token).int()
mask
```




    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           dtype=torch.int32)




```python
enc_input.masked_fill_(mask==0,-1e9)
enc_input

```




    tensor([          1,          66,         642,          29,         113,
                    454,          30,          10,        2524,           7,
                   3305,           4,       21600,       16245,           2,
            -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
            -1000000000, -1000000000, -1000000000, -1000000000])



# Decoder Masking


```python
dec_input
```




    tensor([   1,   56,  775,  209,   52,   17, 2397,    7, 3369,    0,    0,    3,
               3,    3,    3,    3,    3,    3,    3,    3])




```python
casual_mask = torch.triu(torch.ones((1,20,20)), diagonal=1).type(torch.int)
casual_mask==0
```




    tensor([[[ True, False, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]])




```python
casual_mask.shape
```




    torch.Size([1, 20, 20])




```python
dec_input
```




    tensor([   1,   56,  775,  209,   52,   17, 2397,    7, 3369,    0,    0,    3,
               3,    3,    3,    3,    3,    3,    3,    3])




```python
dec_input!=pad_token
```




    tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True, False, False, False, False, False, False, False, False, False])




```python
casual_mask = torch.triu(torch.ones((1,20,20)),diagonal=1).type(torch.int)
```


```python
casual_mask
```




    tensor([[[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],
           dtype=torch.int32)




```python
casual_mask==0
```




    tensor([[[ True, False, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True, False, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True, False, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True, False,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
              False, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True, False, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True, False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
               True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]])




```python
dec_mask = (dec_input!=pad_token)& (casual_mask==0).int()
```


```python
dec_input
```




    tensor([   1,   56,  775,  209,   52,   17, 2397,    7, 3369,    0,    0,    3,
               3,    3,    3,    3,    3,    3,    3,    3])




```python
dec_input.shape
```




    torch.Size([20])




```python
dec_input.masked_fill(dec_mask==0,-1e9)
```




    tensor([[[          1, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000],
             [          1,          56,         775,         209,          52,
                       17,        2397,           7,        3369,           0,
                        0, -1000000000, -1000000000, -1000000000, -1000000000,
              -1000000000, -1000000000, -1000000000, -1000000000, -1000000000]]])


