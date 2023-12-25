import os
import time
from copy import deepcopy
from math import ceil
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import DataLoader


from .data_utils import distributionDataloader
from .get_data import  get_data
# DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# #device id  of this should be same in client_lib device

def load_data(config):
    trainset, testset = get_data(config)
    # Data distribution for non-custom datasets
    if config['dataset'] != 'CUSTOM':
        datasets = distributionDataloader(config,  trainset, config['datapoints'], config['client_idx'])
        trainloader = DataLoader(datasets, batch_size= config['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=config['batch_size'])
        num_examples = {"trainset": len(datasets), "testset": len(testset)}
    else:
        trainloader = DataLoader(trainset, batch_size= config['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=config['batch_size'])
        num_examples = {"trainset": len(trainset), "testset": len(testset)}

    # Return data loaders and number of examples in train and test datasets
    return trainloader, testloader, num_examples


def flush_memory():
    torch.cuda.empty_cache()

def train_model(net, trainloader, epochs, device, deadline=None):

    """
    Trains a neural network model on a given dataset using SGD optimizer with Cross Entropy Loss criterion.
    Args:
        net: neural network model
        trainloader: PyTorch DataLoader object for training dataset
        epochs: number of epochs to train the model
        deadline: optional deadline time for training

    Returns:
        trained model with the difference between trained model and the received model
    """
    x = deepcopy(net)
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Set the model to training mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Check if the deadline time has been reached
        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break

    # Calculate the difference between the trained model and the received model
    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    return net

def train_fedavg(net, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Federated Averaging (FedAvg) algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Set model to train mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Check if deadline has been reached
        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("Deadline occurred.")
                break

    # Return the trained model
    return net

def train_feddyn(net, trainloader, epochs, device, deadline=None, prev_grads=None):
    """
    Trains a given neural network using the FedDyn algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)
    # prev_grads = None

    if prev_grads is not None:
        prev_grads = prev_grads.to(device)
    else:
        for param in net.parameters():
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1))
                prev_grads.to(device)
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
                prev_grads.to(device)

    criterion = torch.nn.CrossEntropyLoss()
   
    lr = 0.1
    alpha = 0.01

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        inputs,labels = next(iter(trainloader))
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        output = net(inputs)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels

        #Dynamic Regularisation
        lin_penalty = 0.0
        curr_params = None
        for param in net.parameters():
            if not isinstance(curr_params, torch.Tensor):
                curr_params = param.view(-1)
            else:
                curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

        lin_penalty = torch.sum(curr_params * prev_grads)
        loss -= lin_penalty

        quad_penalty = 0.0
        for y, z in zip(net.parameters(), x.parameters()):
            quad_penalty += torch.nn.functional.mse_loss(y.data, z.data, reduction='sum')

        loss += (alpha/2) * quad_penalty
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1) # Clip gradients
        optimizer.step()

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break

    #Calculate the difference between updated model (y) and the received model (x)
    delta = None
    for y, z in zip(net.parameters(), x.parameters()):
        if not isinstance(delta, torch.Tensor):
            delta = torch.sub(y.data.view(-1), z.data.view(-1))
        else:
            delta = torch.cat((delta, torch.sub(y.data.view(-1), z.data.view(-1))),dim=0)

    #Update prev_grads using delta which is scaled by alpha
    prev_grads = torch.sub(prev_grads, delta, alpha = alpha)
    return net, prev_grads

def train_mimelite(net, state, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the MimeLite algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model

    In the case of MimeLite, control_variate is nothing but a state like in case of momentum method
    """
    x = deepcopy(net)

    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(net(images), labels)

            #Compute (full-batch) gradient of loss with respect to net's parameters
            grads = torch.autograd.grad(loss,net.parameters())
            #Update net's parameters using gradients
            with torch.no_grad():
                for param,grad,s in zip(net.parameters(), grads, state):
                    param.data = param.data - lr * ((1-momentum) * grad.data + momentum * s.to(device).data)

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break

    #Compute gradient wrt the received model (x) using the wholde dataset
    data = DataLoader(trainloader.dataset, batch_size = len(trainloader) * trainloader.batch_size, shuffle = True)
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        output = x(images)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels
        gradient_x = torch.autograd.grad(loss,x.parameters())

    return net, gradient_x

def train_mime(net, state, control_variate, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Mime algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)

    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    net.train()
    x.train()
    #control_variate = control_variate.to(DEVICE)
    for epoch in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(net(images), labels)

            #Compute (full-batch) gradient of loss with respect to net's parameters
            grads_y = torch.autograd.grad(loss,net.parameters())

            if epoch == 0:
                output = x(images)
                loss = criterion(output, labels)
                grads_x = torch.autograd.grad(loss,x.parameters())

            #Update net's parameters using gradients
            with torch.no_grad():
                for g_y, g_x, c in zip(grads_y, grads_x, control_variate):
                    g_y.data -= g_x.data + c.to(device)

                for param,grad,s in zip(net.parameters(), grads_y, state):
                    param.data = param.data - lr * ((1-momentum) * grad.data + momentum * s.to(device).data)

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break

    #Compute gradient wrt the received model (x) using the wholde dataset
    data = DataLoader(trainloader.dataset, batch_size = len(trainloader) * trainloader.batch_size, shuffle = True)
    for images, labels in data:
        images, labels = images.to(device), labels.to(device)
        output = x(images)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels
        gradient_x = torch.autograd.grad(loss,x.parameters())

    return net, gradient_x

def train_scaffold(net, server_c, trainloader, epochs, device, deadline=None):
    """
    Trains a given neural network using the Scaffold algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model

    """
    x = deepcopy(net)
    client_c = deepcopy(server_c)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(net(images), labels)

            #Compute (full-batch) gradient of loss with respect to net's parameters
            grads = torch.autograd.grad(loss,net.parameters())

            #Update y's parameters using gradients, client_c and server_c [Algorithm line no:10]

            for param,grad,s_c,c_c in zip(net.parameters(),grads,server_c,client_c):
                s_c, c_c = s_c.to(device), c_c.to(device)
                param.data = param.data - lr * (grad.data + (s_c.data - c_c.data))

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break

    delta_c = [torch.zeros_like(param) for param in net.parameters()]
    new_client_c = deepcopy(delta_c)

    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    a = (ceil(len(trainloader.dataset) / trainloader.batch_size) * epochs * lr)
    for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, net.parameters()):
        c_l = c_l.to(device)
        c_g = c_g.to(device)
        n_c.data += c_l.data - c_g.data - diff.data / a

    #Calculate delta_c which equals to new_client_c-client_c
    for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
        d_c = d_c.to(device)
        c_l = c_l.to(device)
        d_c.data.add_(n_c_l.data - c_l.data)


    return net, delta_c

def train_sentiment(net, trainloader, epochs, device, deadline=None):
    
    X_train_sparse = torch.tensor(trainloader['data'], dtype=torch.float32)
    y_train_tensor = torch.tensor(trainloader['target'], dtype=torch.float32)

    
    class SentimentAnalysisModel(nn.Module):
        def __init__(self, input_dim):
            super(SentimentAnalysisModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x
    input_dim = X_train_sparse.shape[1]
    sentiment_model = SentimentAnalysisModel(input_dim)
    sentiment_optimizer = optim.Adam(sentiment_model.parameters(), lr=0.001)
    sentiment_criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(sentiment_optimizer, step_size=10, gamma=0.5)
    class GlobalModel(nn.Module):
        def __init__(self, input_dim):
            super(GlobalModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x 


    sentiment_model = SentimentAnalysisModel(input_dim)
    sentiment_optimizer = optim.Adam(sentiment_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(sentiment_optimizer, step_size=10, gamma=0.5)
    input_dim = 15000
    global_model = GlobalModel(input_dim)

        # Simulate multiple devices (local clients) with local data
    num_devices = 30
    local_models = [GlobalModel(input_dim) for _ in range(num_devices)]
    criterion = nn.BCELoss()
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)
    num_epochs_centralized = 50

    for epoch in range(num_epochs_centralized):
        for device_id in range(num_devices):
            # Simulate local training on each device
            local_optimizer = optim.Adam(local_models[device_id].parameters(), lr=0.01)
            local_data = torch.randn(100, input_dim)  # Simulated local data
            local_labels = torch.randint(0, 2, (100, 1), dtype=torch.float32)  # Simulated labels
            for local_epoch in range(10):  # Local training epochs
                local_optimizer.zero_grad()
                local_outputs = local_models[device_id](local_data)
                local_loss = sentiment_criterion(local_outputs, local_labels)
                local_loss.backward()
                local_optimizer.step()

        # Send local model parameters (or gradients) to the centralized server
        global_model_parameters = global_model.state_dict()
        local_model_parameters = local_models[device_id].state_dict()

        for key in global_model_parameters.keys():
            global_model_parameters[key] += local_model_parameters[key]

        # Average the global model parameters
    for key in global_model_parameters.keys():
        global_model_parameters[key] /= num_devices

        # Update the global model with the averaged parameters
    global_model.load_state_dict(global_model_parameters)
    num_communication_rounds = 15
    for round_num in range(num_communication_rounds):
        print(f"Communication Round {round_num + 1}/{num_communication_rounds}")

        for device_id in range(num_devices):
            # Simulate local training on each device
            local_optimizer = optim.Adam(local_models[device_id].parameters(), lr=0.01)
            local_data = torch.randn(100, input_dim)  # Simulated local data
            local_labels = torch.randint(0, 2, (100, 1), dtype=torch.float32)  # Simulated labels
            for local_epoch in range(10):  # Local training epochs
                local_optimizer.zero_grad()
                local_outputs = local_models[device_id](local_data)
                local_loss = criterion(local_outputs, local_labels)
                local_loss.backward()
                local_optimizer.step()

            # Send local model parameters (or gradients) to the centralized server
            global_model_parameters = global_model.state_dict()
            local_model_parameters = local_models[device_id].state_dict()

            for key in global_model_parameters.keys():
                global_model_parameters[key] += local_model_parameters[key]

        # Average the global model parameters
        for key in global_model_parameters.keys():
            global_model_parameters[key] /= num_devices

        # Update the global model with the averaged parameters
        global_model.load_state_dict(global_model_parameters)
        global_model.eval() 
        with torch.no_grad(): 
            correct = 0 
            total = 0 
            for device_id in range(num_devices):

        # Simulate testing data on each device
                test_data = torch.randn(100, input_dim)  # Simulated testing data
                test_labels = torch.randint(0, 2, (100, 1), dtype=torch.float32)  # Simulated testing labels
                test_outputs = global_model(test_data)
                predicted = (test_outputs > 0.5).float()
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        accuracy = 100 * correct / total 
        print(f'Federated Model Accuracy on the test set: {accuracy:.2f}%')

        
def test_model(net, testloader, device):
    """Evaluate the performance of a model on a test dataset.

    Args:
    net (torch.nn.Module): The neural network model to evaluate.
    testloader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
    Tuple: The average loss and accuracy of the model on the test dataset.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(testloader.dataset)
        accuracy = correct / total
        return test_loss, accuracy
