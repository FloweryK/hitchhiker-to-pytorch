from funcs import onehot, savefig
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Configurations
N_BATCH = 100
N_CLASSES = 10
D_INPUT = 28*28
D_NOISE = 128
N_EPOCH = 200
lr = 0.001


# Before downloading MNIST, Images should have specified format from [0,1] to [-1,1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(dataset=trainset, batch_size=N_BATCH, shuffle=True, num_workers=2)
testloader = DataLoader(dataset=testset, batch_size=N_BATCH, shuffle=True, num_workers=2)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(D_INPUT + N_CLASSES, 512),   # input은 MNIST 그림  + label을 받는다.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(D_NOISE + N_CLASSES, 512),   # input은 Noise + label 을 받는다
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, D_INPUT),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.model(x)
        return out


# define networks
D = Discriminator()
G = Generator()

# define losses and optimizers
criterion = nn.BCELoss()
D_optim = Adam(params=D.parameters(), lr=lr)
G_optim = Adam(params=G.parameters(), lr=lr)

for epoch in range(N_EPOCH):
    for i, (images, labels) in enumerate(trainloader):
        # prepare images, noises for fake images, and real labels.
        images = images.view(-1, D_INPUT)
        noises = torch.randn(images.size(0), D_NOISE)
        labels = onehot(labels, N_CLASSES)

        # concat inputs with labels
        images = torch.cat((images, labels), 1)
        noises = torch.cat((noises, labels), 1)

        # make fake images from generator
        fakes = torch.cat((G(noises), labels), 1)

        # Put real images and fake images to the discriminator
        D_optim.zero_grad()
        D_images = D(images)
        D_fakes = D(fakes)

        # Evaluate discriminator losses and optimize (update network)
        D_image_loss = criterion(D_images, torch.ones(images.size(0)))
        D_fake_loss = criterion(D_fakes, torch.zeros(images.size(0)))
        D_loss = D_image_loss + D_fake_loss
        D_loss.backward()
        D_optim.step()

        # Train the generator
        G_optim.zero_grad()
        G_noises = G(noises)
        G_fakes = torch.cat((G_noises, labels), 1)

        # Evaluate generator losses and optimize (update network)
        G_loss = criterion(D(G_fakes), torch.ones(images.size(0)))
        G_loss.backward()
        G_optim.step()

        # Print the process
        if (i+1) % 10 == 0:
            print('epoch=%i, i=%i, D_loss=%.2f, G_loss=%.2f, D(X)=%.2f, D(G(X))=%.2f'
                  % (epoch, i+1, D_loss.item(), G_loss.item(), D_images.data.mean().item(), D_fakes.data.mean().item()))

    # Let's see what happens during the process
    test_noises = torch.randn(10, D_NOISE)
    test_noises = torch.cat((test_noises, torch.eye(10)), 1)
    samples = G(test_noises)
    savefig(epoch=epoch, samples=samples)

