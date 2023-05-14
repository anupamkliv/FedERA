import os
import sys
import time
import json

from torch.multiprocessing import Process
from torch import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federa.server.src.server import server_start
from federa.client.src.client import client_start

def get_config(action, action2, config_path=""):
    """
    Get the configuration file as json from it 
    """
    
    root_path = os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'configs')
    action = action + '.json'
    with open(os.path.join(config_path, action), encoding='UTF-8') as f1:
        config = json.load(f1)
        config = config[action2]

    return config


def execute(process):
    os.system(f'{process}')

    
def tester(config , no_of_clients, late=None):
    """
    Return the tester to each test algorithm.
    Late is introduced for intermediate connection
    """
    
    multiprocessing.set_start_method('spawn', force=True)
    if late:
        no_of_clients -= 1
    server = Process(target=server_start, args=(config['server'],))
    clients = []
    server.start()
    time.sleep(5)
    for i in range(no_of_clients):
        client = Process(target=client_start, args=(config['client'],))
        clients.append(client)
        client.start()
        time.sleep(2)
    if late:
        time.sleep(3)
        client = Process(target=client_start, args=(config['client'],))
        clients.append(client)
        client.start()
    clients_list = list(range(len(clients)))
    for i in clients_list:
        clients[i].join()
    server.join()

    
def get_result(dataset, algorithm):
    """
    Return the result to each test algorithm.
    Dataset and algorithm defines as for which dataset the result is required
    """
    
    dir_path = './server_results/'+dataset+'/'+algorithm
    lst = os.listdir(dir_path)
    lst.sort()
    lst = lst[-1]
    dir_path = dir_path+'/'+lst
    lst = os.listdir(dir_path)
    lst.sort()
    lst = lst[-1]
    print(lst)
    with open (f'{dir_path}/{lst}/FL_results.txt', 'r', encoding='UTF-8') as file:
        for line in file:
            pass
        result_dict = eval(line)
    return result_dict
    