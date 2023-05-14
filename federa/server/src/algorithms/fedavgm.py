import functools
from collections import OrderedDict

#averages all of the given state dicts
class fedavgm():

    def __init__(self, config):
        self.algorithm = "FedAvgM"
        self.momentum = 0.9
        self.lr = 1
        self.velocity = None

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        #Updates the velocity
        if self.velocity: #This will be False at the first round
            for key in keys:
                self.velocity[key] = self.momentum * self.velocity[key] + avg_delta_y[key]
        else:
            self.velocity = avg_delta_y

        #Uses Nesterov gradient
        for key in keys:
            avg_delta_y[key] += self.momentum * self.velocity[key]

        #Updates server_state_dict
        for key in keys:
            server_state_dict[key] += self.lr * avg_delta_y[key]

        return server_state_dict
