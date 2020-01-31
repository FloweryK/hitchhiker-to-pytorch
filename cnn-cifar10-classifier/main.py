import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            # conv2d(in_channels, out_channels, kernel_size)
            nn.Conv2d(3, 6, 5),                     # 3*(32*32) -> 6*(28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6*(28*28) -> 6*(14*14)
            nn.Conv2d(6, 16, 5),                    # 6*(14*14) -> 16*(10*10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*(10*10) -> 16*(5*5)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.linear_layer(x)
        return x


# image size = 3 channel * 32 * 32
# Images of CIFAR10 have range of [0, 1]. Let's normalize into [-1, 1].
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# there are ten classes
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
net = Net()

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # get outputs
        optimizer.zero_grad()
        outputs = net(inputs)

        # evaluate loss and optimize (update network)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print running loss
        running_loss += loss.data.item()
        if (i+1) % 2000 == 1999:
            print('[%d, %5d (%.2f%%)] loss: %.3f' % (epoch + 1, i + 1, 100 * (i + 1) / len(trainloader), running_loss / 2000))
            running_loss = 0.0

            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                outputs = net(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

