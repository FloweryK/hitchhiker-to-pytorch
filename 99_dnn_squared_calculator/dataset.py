import numpy as np
from torch.utils.data import Dataset


class Customset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.X = np.random.rand(self.size, 1)
        self.T = self.X**2

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.X[item], self.T[item]

