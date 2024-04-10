from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super(SimpleDataset, self).__init__()
        self.x = x.clone().detach().type(torch.float32)
        self.y = y.clone().detach().type(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]