import functools
from collections import OrderedDict

#averages all of the given state dicts
class scaffold():

    def __init__(self, config):
        self.algorithm = "SCAFFOLD"
        self.lr = 1.0
        self.fraction = config["fraction_of_clients"]

    def aggregate(self,server_model_state_dict, control_variate, state_dicts, updated_control_variates):

        keys = server_model_state_dict.keys() #List of keys in a state_dict
        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        delta_x = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            delta_x[key] = current_key_average

        #Average all the  in gradients_x
        delta_c = []
        for i in range(len(control_variate)):
            #Average all the i'th element of updated_control_variate present in the updated_control_variates
            current_tensors = [updated_control_variate[i] for updated_control_variate in updated_control_variates]
            current_sum = functools.reduce(lambda accumulator, tensor: accumulator + tensor, current_tensors)
            current_average = current_sum / len(updated_control_variates)
            delta_c.append(current_average)


        for key in keys:
            server_model_state_dict[key] += self.lr * delta_x[key]

        control_variate_list = list(range(len(control_variate)))
        for i in control_variate_list:
            control_variate[i] += self.fraction * delta_c[i]

        return server_model_state_dict, control_variate
