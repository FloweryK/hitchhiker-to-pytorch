import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PTBdata
from model import CBOW


def make_context_vector(context, word_to_ix):
    idxs = [[word_to_ix[w] for w in context]]
    return torch.tensor(idxs)


if __name__ == '__main__':
    N_BATCH = 32
    N_EMBED = 100
    N_HIDDEN = 128
    WINDOW = 2

    trainset = PTBdata(path='ptb.train.txt',
                       window=WINDOW,)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=N_BATCH,
                             shuffle=True)

    vocab_size = trainset.vocab_size
    word2idx = trainset.word2idx
    idx2word = trainset.idx2word

    model = CBOW(vocab_size, N_EMBED, N_HIDDEN)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    total = len(trainloader)

    for epoch in range(10000):
        running_loss = 0
        for i, data in enumerate(trainloader):
            targets = data['target']
            contexts = data['context']

            prob = model(contexts)
            loss = criterion(prob, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0:
                print('%i %i %i %f %f' % (epoch, i, total, loss, running_loss))
                test = ['researchers', 'who', 'the', 'workers']
                print(test)
                test = make_context_vector(test, word2idx)
                print(idx2word[torch.argmax(model(test)).item()])
