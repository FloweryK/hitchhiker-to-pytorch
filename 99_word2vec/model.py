import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import PTBdata, collate_fn
from torch.utils.data import DataLoader


class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, window=1):
        super(CBOW, self).__init__()

        self.V, self.H = vocab_size, hidden_size
        self.window = window

        # in layer
        self.in_layer = nn.Linear(self.V, self.H, bias=False)

        # out layer
        self.out_layer = nn.Linear(self.H, self.V, bias=False)

    def forward(self, x):
        # x: context one-hot vector (B, N-2*window, 2*window, V)
        # in layer
        x = self.in_layer(x)    # (B, N-2*window, 2*window, H)
        x = F.relu(x)
        x = x.mean(dim=2)       # (B, N-2*window, H)

        # out layer
        x = self.out_layer(x)   # (B, N-2*window, V)
        x = F.softmax(x, dim=2)

        return x


if __name__ == '__main__':
    batch = 64
    hidden_size = 300

    trainset = PTBdata(window=5)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=batch,
                             shuffle=True,
                             num_workers=1,
                             collate_fn=collate_fn)

    vocab_size = trainset.vocab_size

    # model and optimizer
    model = CBOW(vocab_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    total = len(trainset) // batch

    for epoch in range(1000):
        for i, data in enumerate(trainloader):
            # get batch data
            x, y = data  # batch contexts and targets
            x = x.float()
            y = y.float()

            # zero grad
            optimizer.zero_grad()

            # get output
            predict = model(x)

            # cat all targets and outputs
            predict = predict.view(-1, predict.shape[2])
            y = y.view(-1).to(torch.long)

            # update
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()

            # print(list(model.parameters())[0].grad)
            print('epoch: %i | i: %i (/%i) | loss: %.10f' % (epoch, i, total, loss))
