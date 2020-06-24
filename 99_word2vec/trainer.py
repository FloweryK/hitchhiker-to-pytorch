import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PTBdata, collate_fn
from model import CBOW

if __name__ == '__main__':
    trainset = PTBdata(window=1)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=3,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=collate_fn)

    vocab_size = trainset.vocab_size
    hidden_size = 128

    # model and optimizer
    model = CBOW(vocab_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0)

    for data in trainloader:
        # get batch data
        x, y = data     # batch contexts and targets
        x = x.float()
        y = y.float()

        # zero grad
        optimizer.zero_grad()

        # get output
        predict = model(x)
        print(predict.shape)
        print(y.shape)
        print(y)

        # update
        loss = criterion(predict, y)
        loss.backward()
        optimizer.step()


        break