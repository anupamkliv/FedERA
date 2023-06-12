import os
import sys
import time
import json

from torch.multiprocessing import Process
from torch import multiprocessing
from ..server.src.server import server_start
from ..client.src.client import client_start

def get_config(action, action2, config_path=""):
    """
    Get the configuration file as json from it 
    """
    
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, '')
    action = action + '.json'
    with open(os.path.join(config_path, action), encoding='UTF-8') as f1:
        config = json.load(f1)
        config = config[action2]

    return config

def tester(configs , no_of_clients, late=None):
    """
    Return the tester to each test algorithm.
    Late is introduced for intermediate connection
    """
    
    multiprocessing.set_start_method('spawn', force=True)
    if late:
        no_of_clients -= 1
    server = Process(target=server_start, args=(configs['server'],))
    clients = []
    server.start()
    time.sleep(5)
    for i in range(no_of_clients):
        client = Process(target=client_start, args=(configs['client'],))
        clients.append(client)
        client.start()
        time.sleep(2)
    if late:
        time.sleep(3)
        client = Process(target=client_start, args=(configs['client'],))
        clients.append(client)
        client.start()
    clients_list = list(range(len(clients)))
    for i in clients_list:
        clients[i].join()
    server.join()
    