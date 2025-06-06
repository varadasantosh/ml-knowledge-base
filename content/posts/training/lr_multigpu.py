
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
    # print(f"MASTER ADDRESS:- {os.environ.get("MASTER_ADDR")}" )
    print(f"MASTER PORT:- {os.environ.get("MASTER_PORT")}" )
    print(f"World Size:- {os.environ.get("WORLD_SIZE")}")
    global_rank = os.environ.get("RANK")
    epoch = 100
    batch_size =64
    main(int(global_rank),epoch,batch_size)

    # world_size= torch.cuda.device_count()
    # process=[]
    # for rank in range(world_size):
    #     print("Launching new Process on the rank :-", rank)
    #     p =mp.Process(target=main, args=(rank, world_size, 100, 100))
    #     p.start()
    #     process.append(p)
    # for p in process:
    #     p.join()    
    # mp.spawn(main, args=(world_size,500, 64),nprocs=world_size)