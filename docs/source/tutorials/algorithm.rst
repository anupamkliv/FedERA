.. _algorithm:

*****************************
Federated Learning Algorithms
*****************************

The implementation of federated learning algorithms in Feder consists of two components: the training part on the client side and the aggregation part on the server side. The training functions are coded in the net_lib.py file at client/src directory, while the aggregation functions are located in various files within the algorithms folder at server/src directory.

The algorithms currently implemented in **Feder** are:

* FedAvg
* FedDyn
* FedAdam
* FedAdagrad
* Scaffold
* FedAvgM
* Mime
* Mimelite
* FedYogi

Adding a new algorithm to **Feder**
-----------------------------------

To add a new algorithm to **Feder**, you need to implement the training function on the client side and the aggregation function on the server side. The training function should be implemented in the net_lib.py file at client/src directory. The aggregation function should be implemented in a new file in the algorithms folder at server/src directory.

Implementing the training function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training function should be implemented in the net_lib file, in a fashion similar to the following example of the mimelite algorithm:

.. code-block:: python

    def train_mimelite(net, state, trainloader, epochs, deadline=None):
    #In the case of MimeLite, control_variate is nothing but a state like in case of momentum method
    x = deepcopy(net)
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001
    momentum = 0.9
    net.train()

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(net(images), labels)
            
            #Compute (full-batch) gradient of loss with respect to net's parameters 
            grads = torch.autograd.grad(loss,net.parameters())
            #Update net's parameters using gradients
            with torch.no_grad():
                for param,grad,s in zip(net.parameters(), grads, state):
                    param.data = param.data - lr * ((1-momentum) * grad.data + momentum * s.data)

        if deadline:
            current_time = time.time()
            if current_time >= deadline:
                print("deadline occurred.")
                break               
    
    #Compute gradient wrt the received model (x) using the wholde dataset
    data = DataLoader(trainloader.dataset, batch_size = len(trainloader) * trainloader.batch_size, shuffle = True)  
    for images, labels in data:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = x(images)
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels            
        gradient_x = torch.autograd.grad(loss,x.parameters())
    
    return net, gradient_x     

After making the changes in the net_lib.py file, the client_lib.py file also needs to be updated so as to incorporate the newly defined algorithm. The client_lib.py file is located at client/src directory. The following code snippet shows the train function that needs to be updated in the client_lib.py file:

.. code-block:: python

    def train(train_order_message):
        data_bytes = train_order_message.modelParameters
        data = torch.load( BytesIO(data_bytes), map_location="cpu" )
        model_parameters, control_variate, control_variate2 = data['model_parameters'], data['control_variate'], data['control_variate2']
        
        config_dict_bytes = train_order_message.configDict
        config_dict = json.loads( config_dict_bytes.decode("utf-8") )
        carbon_tracker = config_dict["carbon_tracker"]

        model = get_net(config= config_dict)
        model.load_state_dict(model_parameters)
        model = model.to(device)
        epochs = config_dict["epochs"]
        if config_dict["timeout"]:
            deadline = time.time() + config_dict["timeout"]
        else:
            deadline = None
        
        #Run code carbon if the carbon-tracker flag is True
        if (carbon_tracker==1):
            tracker = OfflineEmissionsTracker(country_iso_code="IND", output_dir = save_dir_path)
            tracker.start()
                
        trainloader, testloader, _ = load_data(config_dict)
        print("training started")
        if (config_dict['algorithm'] == 'mimelite'):
            model, control_variate = train_mimelite(model, control_variate, trainloader, epochs, deadline)
        elif (config_dict['algorithm'] == 'scaffold'):
            model, control_variate = train_scaffold(model, control_variate, trainloader, epochs, deadline)
        elif (config_dict['algorithm'] == 'mime'):
            model, control_variate = train_mime(model, control_variate, control_variate2, trainloader, epochs, deadline)
        elif (config_dict['algorithm'] == 'fedavg'):
            model = train_fedavg(model, trainloader, epochs, deadline)
        elif (config_dict['algorithm'] == 'feddyn'):
            model = train_feddyn(model, trainloader, epochs, deadline)
        else:
            model = train_model(model, trainloader, epochs, deadline)
        print("training finished")

        if (carbon_tracker==1):
            emissions: float = tracker.stop()
            print(f"Emissions: {emissions} kg")

        myJSON = json.dumps(config_dict)
        json_path = save_dir_path + "/config.json"
        with open(json_path, "w") as jsonfile:
            jsonfile.write(myJSON)
        json_path = "config.json"
        with open(json_path, "w") as jsonfile:
            jsonfile.write(myJSON)
        
        trained_model_parameters = model.state_dict()
        #Create a dictionary where model_parameters and control_variate are stored which needs to be sent to the server
        data_to_send = {}
        data_to_send['model_parameters'] = trained_model_parameters
        data_to_send['control_variate'] = control_variate #If there is no control_variate, this will become None
        buffer = BytesIO()
        torch.save(data_to_send, buffer)
        buffer.seek(0)
        data_to_send_bytes = buffer.read()   

        print("train eval")
        train_loss, train_accuracy = test_model(model, testloader)
        response_dict = {"train_loss": train_loss, "train_accuracy": train_accuracy}
        response_dict_bytes = json.dumps(response_dict).encode("utf-8")

        train_response_message = TrainResponse(
            modelParameters = data_to_send_bytes, 
            responseDict = response_dict_bytes)

        save_model_state(model)
    return train_response_message

Implementing the aggregation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aggregation function should be implemented within a class in a new file in the algorithms folder at server/src directory. The following code snippet shows the aggregation function for the mimelite algorithm as deffined in the mimelite.py file:

.. code-block:: python

    class mimelite():

        def __init__(self, config):
            self.algorithm = "MimeLite"
            self.lr = 1.0
            self.momentum = 0.9
        
        def aggregate(self,server_model_state_dict, optimizer_state, state_dicts, gradients_x):
            
            keys = server_model_state_dict.keys() #List of keys in a state_dict   
            
            avg_y = OrderedDict() #This will be our new server_model_state_dict
            for key in keys:
                current_key_tensors = [state_dict[key] for state_dict in state_dicts]
                current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
                current_key_average = current_key_sum / len(state_dicts)
                avg_y[key] = current_key_average
                
            #Average all the gradient_x in gradients_x
            avg_grads = []
            for i in range(len(gradients_x[0])):
                #Average all the i'th element of gradient_x present in the gradients_x
                current_tensors = [gradient_x[i] for gradient_x in gradients_x]
                current_sum = functools.reduce(lambda accumulator, tensor: accumulator + tensor, current_tensors)
                current_average = current_sum / len(gradients_x)
                avg_grads.append(current_average)
                
            for state, grad in zip(optimizer_state, avg_grads):
                state.data = self.momentum * state.data + (1 - self.momentum) * grad.data
                
            return avg_y, optimizer_state

