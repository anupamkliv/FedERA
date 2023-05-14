import functools
from collections import OrderedDict

#averages all of the given state dicts
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
