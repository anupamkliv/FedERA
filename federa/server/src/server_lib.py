import torch
import os
from tqdm import tqdm
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from torch.utils import data
import numpy as np
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#serverlib and eval_lib should be on the same device

def load_data(config):
    testset = get_data(config)
    testloader = DataLoader(testset, batch_size=config['batch_size'])
    num_examples = {"testset": len(testset)}
    return testloader, num_examples

### Load different dataset
def get_data(config):
    dataset_path="./server_dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if config['dataset'] == 'MNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.MNIST(root='./server_dataset/MNIST',
                                train=False, download=True, transform=apply_transform)
    if config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.FashionMNIST(root='./server_dataset/FashionMNIST',
                                        train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.CIFAR10(root='./server_dataset/CIFAR10',
                                   train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = datasets.CIFAR100(root='./server_dataset/CIFAR100',
                                    train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CUSTOM':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        testset = customDataset(root='./server_custom_dataset/CUSTOM/test', transform=apply_transform)

    return testset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        samples = sample_return(root)

        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        img, label= self.samples[index]

        img = np.load(img)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.samples)

def sample_return(root):
    newdataset = []
    labels = {'Breast': 0, 'Chestxray':1, 'Oct': 2, 'Tissue': 3}
    for image in os.listdir(root):
        label=[]
        #print(image)
        path = os.path.join(root, image)
        #print(path)
        labels_str = image.split('_')[0]
        label = labels[labels_str]
        item = (path, label)
        newdataset.append(item)
    return newdataset



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

def train_model(net, trainloader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    outputs = net(images)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return net

def test_model(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader) :
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def save_intial_model(config):
    testloader, _ = load_data(config)
    net = get_net(config)
    net = train_model(net, testloader)
    torch.save(net.state_dict(), 'initial_model.pt')
