
import random
import threading
from math import ceil

#holds references to all live client_wrapper objects
class ClientManager:
    def __init__(self):
        self.client_list = []
        self.cv = threading.Condition()
        self.accepting_connections = True #set to false to stop accepting further connections

    #returns a list of references to client wrapper objects in the order they connected
    def select(self, num_of_clients = None, fraction = None, timeout = None):
        if num_of_clients:
            self.wait_for(num_of_clients, timeout = timeout)
        if num_of_clients and fraction:
            num_of_clients = ceil( fraction * num_of_clients )
        if num_of_clients is None and fraction:
            num_of_clients = ceil( fraction * len(self.client_list) )
        if num_of_clients is None and fraction is None:
            num_of_clients = len(self.client_list)
        selected_clients_list = self.client_list[:num_of_clients]
        return selected_clients_list

    #same as select but random order
    def random_select(self, num_of_clients = None, fraction = None, timeout = None):
        if num_of_clients:
            self.wait_for(num_of_clients, timeout = timeout)
        if num_of_clients and fraction:
            num_of_clients = ceil( fraction * num_of_clients )
        if num_of_clients is None and fraction:
            num_of_clients = ceil( fraction * len(self.client_list) )
        if num_of_clients is  None and fraction is None:
            num_of_clients = len(self.client_list)
        client_list = self.client_list
        if len(client_list) < num_of_clients:
            return client_list
        selected_clients_list = random.sample(client_list, k=num_of_clients)
        return selected_clients_list


    #used to add a client wrapper object when accepting a new connection
    def register(self, client):
        if not self.accepting_connections:
            return False
        with self.cv:
            self.client_list.append(client)
            self.cv.notify_all()
        return True

    def num_connected_clients(self):
        return len(self.client_list)

    def deregister(self, client_index):
        self.client_list.pop(client_index)

    #wait for the number of clients to connect, indefinitely.
    # unless a timeout is specified, then just return after timeout
    def wait_for(self, minimum_clients, timeout):
        with self.cv:
            self.cv.wait_for( lambda: len(self.client_list) >= minimum_clients, timeout )
            