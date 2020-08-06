import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, N_EMBED):
        super(CBOW, self).__init__()

        # embedding layer (B, 2*window) -> (B, 2*window, N_EMBED)
        self.embedding = nn.Embedding(vocab_size, N_EMBED)

        # out layer (B, 2*window, N_EMBED) -> (B, 2*window, vocab_size)
        self.linear = nn.Linear(N_EMBED, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: context (B, 2*window)
        x = self.embedding(x)       # (B, 2*window, N_EMBED)
        x = x.mean(dim=1)           # (B, N_EMBED)
        x = self.linear(x)          # (B, vocab_size)
        x = self.softmax(x)         # (B, vocab_size)
        return x


class CBOW_ngs(nn.Module):
    def __init__(self, vocab_size, N_EMBED):
        super(CBOW_ngs, self).__init__()

        # embedding layer (B, 2*window) -> (B, 2*window, N_EMBED)
        self.embedding = nn.Embedding(vocab_size, N_EMBED)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x: context (B, 2*window)
        # y: target (B, )
        x = self.embedding(x)       # x: (B, 2*window, N_EMBED)
        x = x.mean(dim=1)           # x: (B, N_EMBED)
        y = self.embedding(y)       # y: (B, N_EMBED)

        return x
