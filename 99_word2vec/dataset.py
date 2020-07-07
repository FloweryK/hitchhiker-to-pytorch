import re
import torch
from torch.utils.data import Dataset, DataLoader


class PTBdata(Dataset):
    def __init__(self, path='ptb.train.txt', window=2, limit=0):
        self.path = path
        self.limit = limit
        self.window = window
        self.targets, self.contexts, self.word2idx, self.idx2word = self.preprocess(mode='not test')
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {
            'target': torch.tensor(self.targets[i]),
            'context': torch.tensor(self.contexts[i])
        }

    def preprocess(self, mode='test'):
        if mode is 'test':
            raw_text = """We are about to study the idea of a computational process.
            Computational processes are abstract beings that inhabit computers.
            As they evolve, processes manipulate other abstract things called data.
            The evolution of a process is directed by a pattern of rules
            called a program. People create programs to direct processes. In effect,
            we conjure the spirits of the computer with our spells."""

            sentences = [sentence.split() for sentence in raw_text.split('.')]
        else:
            # read lines and preprocess
            sentences = []
            with open(self.path, 'r') as f:
                while True:
                    # read line
                    sentence = f.readline()
                    if not sentence:
                        break

                    # convert one sentence(string) into words(list)
                    sentence = re.compile("[a-zA-Z'.#$]+|<unk>").findall(sentence)

                    # add sentence if only the size is enough
                    if len(sentence) >= 1 + 2 * self.window:
                        sentences.append(sentence)

                    # load limit
                    if self.limit and (len(sentences) >= self.limit):
                        break

        # make vocab and dictionary
        vocab = sorted(list(set().union(*sentences)))
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}

        # make targets and contexts
        targets = []
        contexts = []
        for sentence in sentences:
            for i in range(self.window, len(sentence) - self.window):
                # get target and context
                target = sentence[i]
                context = sentence[i-self.window:i] + sentence[i+1:i+1+self.window]

                # convert into indice representation
                target = word2idx[target]
                context = [word2idx[word] for word in context]

                # append results
                targets.append(target)
                contexts.append(context)

        return targets, contexts, word2idx, idx2word


if __name__ == '__main__':
    trainset = PTBdata(path='ptb.train.txt', window=2)
    trainloader = DataLoader(dataset=trainset, batch_size=3)

    for data in trainloader:
        print(data['target'])
        print(data['context'])



