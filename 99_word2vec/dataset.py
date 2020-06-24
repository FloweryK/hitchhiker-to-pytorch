import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class PTBdata(Dataset):
    def __init__(self, path='ptb.train.txt', window=1, limit=0):
        self.path = path
        self.window = window
        self.limit = limit

        self.sentences, self.word2idx, self.idx2word = self.preprocess()
        self.size = len(self.sentences)
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        def _get_contexts(sentence, window):
            return [sentence[i:i+window] + sentence[i+window+1:i+2*window+1] for i in range(len(sentence)-2*window)]

        def _get_targets(sentence, window):
            return [sentence[i+window] for i in range(len(sentence)-2*window)]

        # extract contexts and targets from a single sentence
        sentence = self.sentences[item]                     # (1, N)
        contexts = _get_contexts(sentence, self.window)     # (N-2*window, 2*window)
        targets = _get_targets(sentence, self.window)       # (1, N-2*window)

        # convert idx representations into one-hot representation
        contexts = F.one_hot(torch.tensor(contexts), self.vocab_size)   # (N-2*window, 2*window, vocab_size)

        # targets remain idx representations for cross-entropy loss in pytorch
        targets = torch.tensor(targets)

        return contexts, targets

    def preprocess(self):
        # read lines and preprocess
        sentences = []
        with open(self.path, 'r') as f:
            while True:
                # read line
                sentence = f.readline()
                if not sentence:
                    break

                if self.limit and (self.limit < len(sentences)):
                    break

                # convert one sentence(string) into words(list)
                sentence = re.compile("[a-zA-Z'.#$]+|<unk>").findall(sentence)

                # add sentence if only the size is enough
                if len(sentence) >= 1 + 2 * self.window:
                    sentences.append(sentence)

        # make vocab and dictionary
        vocab = sorted(list(set().union(*sentences)))
        vocab = ['<PADDING>'] + vocab   # padding index: 0
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}

        # convert sentences into index representations
        sentences = [[word2idx[word] for word in sentence] for sentence in sentences]

        return sentences, word2idx, idx2word


def collate_fn(batch):
    max_len = max([len(targets) for _, targets in batch])

    # add padding and stack
    batch_contexts = torch.stack([F.pad(contexts, [0, 0, 0, 0, 0, max_len-contexts.shape[0]]) for contexts, _ in batch])
    batch_targets = torch.stack([F.pad(targets, [0, max_len-targets.shape[0]]) for _, targets in batch])

    return batch_contexts, batch_targets


if __name__ == '__main__':
    trainset = PTBdata(window=2)
    trainloader = DataLoader(dataset=trainset,
                             batch_size=3,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=collate_fn)

    for data in trainloader:
        print(data[0])
        print(data[1])
        print(data[0].shape)
        print(data[1].shape)
        break



