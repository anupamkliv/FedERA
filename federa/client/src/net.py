from torch import nn
from torchvision import models

class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x

def get_net(config):
    if config["net"] == 'LeNet':
        if config['dataset'] in ['MNIST', 'FashionMNIST', 'CUSTOM']:
            net = LeNet(in_channels=1, num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = LeNet(in_channels=3, num_classes=10)
        else:
            net = LeNet(in_channels=3, num_classes=100)
    if config["net"] == 'resnet18':
        if config['dataset'] in ['MNIST', 'FashionMNIST']:
            net = models.resnet18(num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = models.resnet18(num_classes=10)
        else:
            net = models.resnet18(num_classes=100)
    if config["net"] == 'resnet50':
        if config['dataset'] in ['MNIST', 'FashionMNIST']:
            net = models.resnet50(num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = models.resnet50(num_classes=10)
        else:
            net = models.resnet50(num_classes=100)
    if config["net"] == 'vgg16':
        if config['dataset'] in ['MNIST', 'FashionMNIST']:
            net = models.vgg16(num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = models.vgg16(num_classes=10)
        else:
            net = models.vgg16(num_classes=100)
    if config['net'] == 'AlexNet':
        if config['dataset'] in ['MNIST', 'FashionMNIST']:
            net = models.alexnet(num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = models.alexnet(num_classes=10)
        else:
            net = models.alexnet(num_classes=100)
    return net
