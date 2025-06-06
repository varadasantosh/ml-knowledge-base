

---
title: "Pytorch DDP Setup -Multi Node"
---

<!--more-->

##  1️⃣  EC2 Infrastructure Setup

Launch Two EC2 Instances

- **Instance Type**: `g4dn.xlarge`
- **AMI**: Ubuntu 22.04 (x86_64)

**Note** Intially tried with Amazon Linux – there were compatibility issues with NVIDIA libraries.

| Hostname      | Private DNS                                      |
|---------------|--------------------------------------------------|
| pytorch-ddp-1 | `ip-172-31-8-59.us-west-2.compute.internal`      |
| pytorch-ddp-2 | `ip-172-31-9-180.us-west-2.compute.internal`     |


- ✅ Assigned same SSH keypair for both EC2 Instances
- ✅ Assigned the same VPC & Subnet (This is required for both MASTER & WORKER to communicate)
- ✅ Assigned default Security Group
- ✅ Update Seucrity Group Inbound Rules to allow TCP traffic on PORT **22** (Required for connecting to instances via SSH) 
- ✅ Add new Inbound Rule to Security Group to allow port range **0-65535** referring to the same Security Groups as Source
     (Security Group assigned to EC2 Instance)


## 2️⃣  NVIDIA Driver Installation

- sudo apt update
- sudo apt install -y nvidia-driver-535
    
## 3️⃣  Validate Network Communication Between Nodes

- Master -> Worker

``` shell 
ssh -i "pytorch-dpp.pem" ubuntu@ec2-35-88-110-204.us-west-2.compute.amazonaws.com
nc -zv 172.31.9.180 22
```


![master to worker](/images/training/DDP/master_worker.png)

- Worker -> Master 
         
```shell
ssh -i pytorch-dpp.pem ubuntu@ec2-34-221-186-92.us-west-2.compute.amazonaws.com
nc -zv 172.31.8.59 22
```

![worker to master](/images/training/DDP/worker_master.png)

## 4️⃣ Transfer Training Script to Instance

 ```shell 
  scp -i pytorch-dpp.pem lr_multigpu.py ubuntu@ec2-35-88-110-204.us-west-2.compute.amazonaws.com:/home/ubuntu
  scp -i pytorch-dpp.pem utils/dataset.py ubuntu@ec2-35-88-110-204.us-west-2.compute.amazonaws.com:/home/ubuntu/utils
  scp -i pytorch-dpp.pem requirements.txt ubuntu@ec2-35-88-110-204.us-west-2.compute.amazonaws.com:/home/ubuntu
  ```

## 5️⃣ Training Script

``` python 
# lr_multigpu.py


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import datetime

from torch.utils.data import  DataLoader,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
from sklearn.datasets import make_regression
from utils.dataset import MyDataset


class torchMultiLR():

    def __init__(self, model, loss_fn, optimizer, gpu_rank, epoch,features, target):
        print(f"Initialize Training Model , Loss Function, Optimizer etc on GPU Rank: {gpu_rank}")

        self.model = model.to(gpu_rank)
        self.model = DDP(self.model, device_ids=[gpu_rank])
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.gpu_rank = gpu_rank
        self.epoch = epoch
        self.features = features
        self.target = target
        print("Completed Initializing Training Model , Loss Function, Optimizer etc")

    def fit(self, dataloader:DataLoader):
         for epoch in range(self.epoch):
            for batch_idx, (features, target, indices) in enumerate(dataloader):
                local_rank =  int(os.environ.get("LOCAL_RANK", 0))
                print(f"training on epoch:-{epoch} , rank :-{local_rank}")
                print(f" Epoch:- {epoch} [Rank {dist.get_rank()}] Batch {batch_idx} - Sample indices: {indices.tolist()}")
                features =features.to(self.gpu_rank)
                target = target.to(self.gpu_rank)
                y_pred = self.model(features)
                loss = self.loss_fn(y_pred,target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Current Iteration:-{epoch}, on Running on Rank:-{self.gpu_rank}, loss is %.2f", loss)
            self._savechckpt("gpucheckpt.pt")

    def _savechckpt(self, path):

        if self.gpu_rank==0:
            snapshot={}
            snapshot['model_params'] = self.model.module.state_dict()
            snapshot['features'] = self.features.to(self.gpu_rank)
            snapshot['target'] = self.target.to(self.gpu_rank)
            snapshot['model'] = self.model.module
            torch.save(snapshot,path)


def setup_ddp(rank, world_size):
   
    print("Before Initializing Process Group")
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl")
    # init_process_group(backend="nccl",rank=rank, world_size=world_size,timeout=datetime.timedelta(seconds=120))
    print("After Initializing Process Group")

def main(rank:int, epoch:int ,batch_size:int):
        
        local_rank =  int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_ddp(local_rank,world_size)
        data = make_regression(n_samples=100000, n_features=1, n_targets=1, random_state=42)
        print("Data Preparation")
        features = torch.tensor(data[0], dtype=torch.float32).to(local_rank)
        target = torch.tensor(data[1], dtype=torch.float32).to(local_rank)
        target = target.view(target.shape[0],1)
        custom_dataset = MyDataset(features,target)
        custom_dataloader = DataLoader(dataset= custom_dataset,
                                       batch_size= batch_size,
                                       shuffle=False,
                                       sampler= DistributedSampler(custom_dataset)
                                        )
        print("Model Initialized")                                
        model = torch.nn.Sequential(nn.Linear(features.shape[1],5),
                                    nn.Linear(5,1)).to(local_rank)
                                  
        print("Loss Function & Optimizer Definition")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)                                    
        torch_lr = torchMultiLR(model=model,loss_fn=loss_fn,optimizer=optimizer,gpu_rank=local_rank,epoch=epoch, features=features,target=target)
        print("Starting trainign process")
        torch_lr.fit(custom_dataloader)
        destroy_process_group()
        

if __name__ == "__main__":

    print(f"Global Rank:- {os.environ.get("RANK")}")
    print(f"Node Rank:-, {os.environ.get("LOCAL_RANK"),0}")
    print(f"World Size:- {os.environ.get("WORLD_SIZE")}")
    global_rank = os.environ.get("RANK")
    epoch = 100
    batch_size =64
    main(int(global_rank),epoch,batch_size)

    
```


```python
# utils/dataset.py

from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, features, target):
        self.features = features
        self.target  = target

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,index):
        return (self.features[index], self.target[index],index)
```

## 6️⃣  Python Dependencies (`requirements.txt`)
``` text
torch==2.6.0
scikit-learn==1.6.1

```
##  7️⃣  Environment Setup
```shell
 sudo apt install python3.12-venv
 python3 -vnenv ddp
 source ddp/bin/activate
 pip install -r requirements.txt
 export WORLD_SIZE=2
```
## 8️⃣  Running DDP with torchrun
  ▶️ On Master Node (Rank 0)
  
    torchrun --nproc_per_node=1 \
            --nnodes=2 \
            --node-rank=0 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=172.31.8.59:29500 lr_multigpu.py

  ▶️ On Worker Node (Rank 1)           

    torchrun \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node-rank=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=172.31.8.59:29500 \
        lr_multigpu.py  

