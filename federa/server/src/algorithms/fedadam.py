import functools
from collections import OrderedDict
import torch

#averages all of the given state dicts
class fedadam():

    def __init__(self, config):
        self.algorithm = "FedAdam"
        self.lr = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1

        self.m = None #1st moment vectpr
        self.v = None #2nd moment vector

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        if not self.m: #If self.m = None, then the following line will execute. So only at first round, it'll execute
            self.m = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]
            self.v = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]

        #Updates the server_state_dict
        for key, m, v in zip(keys, self.m, self.v):
            m.data = self.beta1 * m.data + (1 - self.beta1) * avg_delta_y[key].data
            v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(avg_delta_y[key].data)
            m_bias_corr = m / (1 - self.beta1**self.timestep)
            v_bias_corr = v / (1 - self.beta2**self.timestep)
            server_state_dict[key].data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)


        self.timestep += 1 #After each aggregation, timestep will increment by 1

        return server_state_dict
