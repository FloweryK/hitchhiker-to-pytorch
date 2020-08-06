import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import PTBdata
from model import CBOW, CBOW_ngs


def make_context_vector(context, word_to_ix):
    idxs = [[word_to_ix[w] for w in context]]
    return torch.tensor(idxs)


# SETTINGS
N_BATCH = 32
N_EMBED = 100
WINDOW = 2
N_EPOCH = 10000


if __name__ == '__main__':
    # dataset, dataloader
    trainset = PTBdata(path='ptb.train.txt',
                       window=WINDOW,
                       limit=100)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=N_BATCH,
                             shuffle=True)

    # make vocab
    vocab_size = trainset.vocab_size
    word2idx = trainset.word2idx
    idx2word = trainset.idx2word

    # model, loss, optimizer
    model = CBOW(vocab_size, N_EMBED)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(N_EPOCH):
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
                print('epoch: %i | i: %i (/%i) | loss: %f (running loss: %f)' % (epoch, i, len(trainloader), loss, running_loss))
                test = ['researchers', 'who', 'the', 'workers']
                print(test)
                test = make_context_vector(test, word2idx)
                print(idx2word[torch.argmax(model(test)).item()])
