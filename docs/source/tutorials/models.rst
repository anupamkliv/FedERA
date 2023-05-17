.. _models:

*******
Models 
*******

The models currently implemented in the framework are:

* LeNet
* ResNet18
* ResNet50
* VGG16
* AlexNet

The `server_lib.py` file contains the implementation of Deep-Learning models for the server, while the `net.py` file contains the implementation of these models for the client. These models are either created by inheriting from torch.nn.module or are imported from torchvision.models.

Adding support for a new model
------------------------------

There are two ways to incorporate support for new models in **FedERA**. One involves creating a new class that inherits from torch.nn.module and the other involves importing a model from torchvision.models. The first method is more flexible and allows for more customization, while the second method is easier to implement and is recommended for beginners.

Adding support for a new model by inheriting from torch.nn.module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add support for a new model by inheriting from torch.nn.module, the following steps need to be followed:

1. Create a new class that inherits from torch.nn.module and defines the model that needs to be implemented, and add it to `server_lib.py` file and the `net.py` file. The code for LeNet is given below as an example:

.. code-block:: python

    class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
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

2. The models implemented in `client_lib.py` and `net.py` files are imported via the use of `get_net.py` function defined in both the files. This function takes in the name of the model as a string and returns the corresponding model. To add support for a new model, the name of the model needs to be added to the `get_net.py` function in both the files and appropriate changes need to be made. The code for the `get_net.py` function in `client_lib.py` is given below as an example:

.. code-block:: python

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

Adding support for a new model by importing from torchvision.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add support for a new model by importing from torchvision.models, import the model from torchvision.models in `server_lib.py` and `net.py` files and make changes in the `get_net` function appropriately. The code that needs to be added in `get_net` function to import ResNet38 model is given below as an example:

.. code-block:: python

    if config["net"] == 'resnet38':
        if config['dataset'] in ['MNIST', 'FashionMNIST']:
            net = models.resnet38(num_classes=10)
        elif config['dataset'] == 'CIFAR10':
            net = models.resnet38(num_classes=10)
        else:
            net = models.resnet38(num_classes=100)




