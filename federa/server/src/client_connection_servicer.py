
from queue import Queue

from . import ClientConnection_pb2_grpc

from .client_wrapper import ClientWrapper

#gRPC servicer that contains all functions that can be called by the client
class ClientConnectionServicer( ClientConnection_pb2_grpc.ClientConnectionServicer ):
    def __init__(self, client_manager):
        self.client_manager = client_manager

    #called by every newly connected client. executes in a different thread for every client.
    #creates a client wrapper object for the client, registers with client manager,
    # and passes message from server to client
    #and vice-versa
    def Connect(self, request_iterator, context):
        client_id = context.peer()
        client_message_iterator = request_iterator
        send_buffer = Queue(maxsize = 1)
        recieve_buffer = Queue(maxsize = 1)

        client = ClientWrapper(send_buffer, recieve_buffer, client_id)
        register_result = self.client_manager.register(client)
        #if server is accepting connections, and registering was successful, True is returned
        if register_result:
            print(f"Client {client_id} connected.")
            client_index = self.client_manager.num_connected_clients() - 1
            client.client_idx = client_index

            try:
                while True:
                    server_message = send_buffer.get()
                    yield server_message
                    client_message = next(client_message_iterator)
                    recieve_buffer.put(client_message)
            finally:
                client.is_connected = False
                self.client_manager.deregister(client_index)
                print(f"Client {client_id} has disconnected.")
                print(f"{self.client_manager.num_connected_clients()} clients remain active.")
        #server is not accepting connections or registering failed
        else:
            client.disconnect()
            server_message = send_buffer.get()
            yield server_message
            print(f"Client {client_id} attempted to connect. Connection refused.")
            