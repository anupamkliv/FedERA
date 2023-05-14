import os
from torchvision import transforms,datasets
from torch.utils import data
import numpy as np
from PIL import Image

# Define a function to get the train and test datasets based on the given configuration
def get_data(config):
    # If the dataset is not custom, create a dataset folder
    if config['dataset'] != 'CUSTOM':
        dataset_path = "client_dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    # Get the train and test datasets for each supported dataset
    if config['dataset'] == 'MNIST':
        # Apply transformations to the images
        apply_transform = transforms.Compose([transforms.Resize(config["resize_size"]), transforms.ToTensor()])
        # Download and load the trainset
        trainset = datasets.MNIST(root='client_dataset/MNIST', train=True, download=True, transform=apply_transform)
        # Download and load the testset
        testset = datasets.MNIST(root='client_dataset/MNIST', train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=True, download=True, transform=apply_transform)
        testset = datasets.FashionMNIST(root='client_dataset/FashionMNIST',
                                        train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                    train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR10(root='client_dataset/CIFAR10',
                                   train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                     train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR100(root='client_dataset/CIFAR100',
                                    train=False, download=True, transform=apply_transform)
    elif config['dataset'] == 'CUSTOM':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        # Load the custom dataset
        trainset = customDataset(root='client_custom_dataset/CUSTOM/train', transform=apply_transform)
        testset = customDataset(root='client_custom_dataset/CUSTOM/test', transform=apply_transform)
    else:
        # Raise an error if an unsupported dataset is specified
        raise ValueError(f"Unsupported dataset type: {config['dataset']}")


    # Return the train and test datasets
    return trainset, testset

class customDataset(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset class for loading image and label data from a folder of .npy files.
        Args:
            root (str): Path to the folder containing the .npy files.
            transform (callable, optional): A function/transform that takes
              an PIL image and returns a transformed version.
                                            E.g, `transforms.RandomCrop`
        """

        self.root = root
        samples = sample_return(root)

        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            img (PIL.Image): The image data.
            label (int): The label for the image data.
        """
        img, label= self.samples[index]

        img = np.load(img)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        return img, label

    def __len__(self):
        return len(self.samples)

def sample_return(root):
    # Initialize an empty list to hold the samples
    newdataset = []
    # Define a dictionary that maps label names to integer values
    labels = {'Breast': 0, 'Chestxray':1, 'Oct': 2, 'Tissue': 3}
    # Loop over each image in the root directory
    for image in os.listdir(root):
        # Initialize an empty list to hold the label
        label=[]
        # Get the full path of the image
        path = os.path.join(root, image)
        # Extract the label from the image filename
        labels_str = image.split('_')[0]
        label = labels[labels_str]
        # Create a tuple containing the image path and its label, and append it to the list of samples
        item = (path, label)
        newdataset.append(item)
    # Return the list of samples
    return newdataset
