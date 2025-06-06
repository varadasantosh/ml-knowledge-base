from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, features, target):
        self.features = features
        self.target  = target

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,index):
        return (self.features[index], self.target[index],index)
