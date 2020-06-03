import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class Customset(Dataset):
    def __init__(self, size=10000):
        self.size = size
        self.X = np.random.rand(self.size, 1)
        self.T = self.X**2

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.X[item], self.T[item]


if __name__ == '__main__':
    # dataset
    trainset = Customset()
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)

    # model and optimizer
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)

    # trainer
    for epoch in range(1000):
        for i, data in enumerate(trainloader):
            # get batch data
            x, t = data
            x = x.float()
            t = t.float()

            # zero grad
            optimizer.zero_grad()

            # get output
            predict = net(x)

            # update
            loss = criterion(predict, t)
            loss.backward()
            optimizer.step()

            print('\n epoch: %i | i: %i | loss: %.5f' % (epoch, i, loss))
            print(t.detach().numpy().tolist())
            print(predict.detach().numpy().tolist())

