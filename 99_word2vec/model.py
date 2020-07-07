import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, N_EMBED, N_HIDDEN):
        super(CBOW, self).__init__()

        # embedding layer (B, 2*window) -> (B, 2*window, N_EMBED)
        self.embedding = nn.Embedding(vocab_size, N_EMBED)

        # out layer (B, 2*window, N_HIDDEN) -> (B, 2*window, vocab_size)
        self.linear2 = nn.Linear(N_EMBED, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: context (B, 2*window)
        # label: (B, )
        x = self.embedding(x)       # (B, 2*window, N_EMBED)
        x = x.mean(dim=1)           # (B, N_EMBED)
        # x = self.linear1(x)         # (B, N_HIDDEN)
        # x = self.relu(x)            # (B, N_HIDDEN)
        x = self.linear2(x)         # (B, vocab_size)
        x = self.softmax(x)         # (B, vocab_size)
        return x
