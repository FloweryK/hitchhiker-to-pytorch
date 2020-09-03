import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from dataset import Customset


def run():
    # Configurations
    N_EPOCH = 1000
    N_BATCH = 10
    lr = 0.1

    # dataset
    trainset = Customset()
    trainloader = DataLoader(trainset, batch_size=N_BATCH, shuffle=True, num_workers=1)

    # model and optimizer
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)

    # trainer
    for epoch in range(N_EPOCH):
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


if __name__ == '__main__':
    run()


