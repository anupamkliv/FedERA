import functools
from collections import OrderedDict

#averages all of the given state dicts
class fedavg():

    def __init__(self, config):
        self.algorithm = "FedAvg"

    def aggregate(self,server_state_dict,state_dicts):
        #server_state_dict is of no use in FedAvg,
        # to maintain consistency with other algorithms; it is provided as an argument
        result_state_dict = OrderedDict()
        for key in state_dicts[0].keys():
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            result_state_dict[key] = current_key_average

        return result_state_dict
