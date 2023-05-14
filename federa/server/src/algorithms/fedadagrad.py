import functools
from collections import OrderedDict
import torch

#averages all of the given state dicts
class fedadagrad():

    def __init__(self, config):
        self.algorithm = "FedAdagrad"
        self.lr = 0.01
        self.epsilon = 1e-6
        self.state = None

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        if not self.state: #If state = None, then the following line will execute.
            #So only at first round, it'll execute
            self.state = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]

        #Updates the server_state_dict
        for key, state in zip(keys, self.state):
            state.data += torch.square(avg_delta_y[key])
            server_state_dict[key] += self.lr * avg_delta_y[key] / torch.sqrt(state.data + self.epsilon)

        return server_state_dict
