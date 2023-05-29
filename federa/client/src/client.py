from queue import Queue
import torch
import time

import grpc
from . import ClientConnection_pb2_grpc
from .ClientConnection_pb2 import ClientMessage

from .client_lib import train, evaluate, set_parameters

def client_start(config):
    keep_going = True
    wait_time = config["wait_time"]
    ip_address = config["ip_address"]
    device = torch.device(config["device"])
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    while keep_going:
        #wait for specified time before reconnecting
        time.sleep(wait_time)
        if config["encryption"] == 1:
            ca_cert = 'ca.pem'
            root_certs = bytes(open(ca_cert).read(), 'utf-8')
            credentials = grpc.ssl_channel_credentials(root_certs)
            #create new gRPC channel to the server
            channel = grpc.secure_channel(ip_address, options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
                ], credentials=credentials)
        else:
            channel = grpc.insecure_channel(ip_address, options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)
                ])
        stub = ClientConnection_pb2_grpc.ClientConnectionStub(channel)
        client_buffer = Queue(maxsize = 10)
        print("Connected with server")

        #wait for incoming messages from the server in client_buffer
        #then according to fields present in them call the appropraite function
        for server_message in stub.Connect( iter(client_buffer.get, None) ):
            if server_message.HasField("evalOrder"):
                eval_order_message = server_message.evalOrder
                eval_response_message = evaluate(eval_order_message, device)
                message_to_server = ClientMessage(evalResponse = eval_response_message)
                client_buffer.put(message_to_server)

            if server_message.HasField("trainOrder"):
                train_order_message = server_message.trainOrder
                train_response_message = train(train_order_message, device)
                message_to_server = ClientMessage(trainResponse = train_response_message)
                client_buffer.put(message_to_server)

            if server_message.HasField("setParamsOrder"):
                set_parameters_order_message = server_message.setParamsOrder
                set_parameters(set_parameters_order_message, device)
                message_to_server = ClientMessage(setParamsResponse = None)
                client_buffer.put(message_to_server)

            if server_message.HasField("disconnectOrder"):
                print("Current FL process is done ")
                disconnect_order_message = server_message.disconnectOrder
                message = disconnect_order_message.message
                print(message)
                reconnect_time = disconnect_order_message.reconnectTime
                if reconnect_time == 0:
                    keep_going = False
                    break
                wait_time = reconnect_time
