
from io import BytesIO
import torch
import json

from .ClientConnection_pb2 import ServerMessage, TrainOrder, EvalOrder, SetParamsOrder, DisconnectOrder

#serves as an abstraction of the actual connected client.
#methods called here are called on the actual client with the same inputs and outputs
class ClientWrapper:
    def __init__(self, send_buffer, recieve_buffer, client_id):
        #data is placed in this buffer to send to client
        self.send_buffer = send_buffer
        #data recieved from client is extracted from this buffer
        self.recieve_buffer = recieve_buffer
        self.client_id = client_id
        self.is_connected = True

    #orders the connected client to train using the given parameters
    def train(self, model_parameters, control_variate, control_variate2, config_dict):
        self.check_disconnection()

        #Create a dictionary where model_parameters and control_variate are stored
        data = {}
        data['model_parameters'] = model_parameters
        data['control_variate'] = control_variate
        data['control_variate2'] = control_variate2

        #convert data to bytes
        buffer = BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        data_bytes = buffer.read()

        #convert config_dict to bytes
        config_dict_bytes = json.dumps(config_dict).encode("utf-8")

        #send bytes to client
        train_order_message = TrainOrder(
            modelParameters = data_bytes,
            configDict = config_dict_bytes)
        message_to_client = ServerMessage(trainOrder = train_order_message)
        self.send_buffer.put(message_to_client)

        #get trained model_parameters and response_dict from client
        client_message = self.recieve_buffer.get()
        train_response_message = client_message.trainResponse
        data_received_bytes = train_response_message.modelParameters
        data_received = torch.load( BytesIO(data_received_bytes), map_location="cpu" )
        #updated_control_variate will become None when no control_variate is involved at all
        trained_model_parameters = data_received['model_parameters']
        updated_control_variate = data_received['control_variate']
        response_dict_bytes = train_response_message.responseDict
        response_dict = json.loads( response_dict_bytes.decode("utf-8") )
        return trained_model_parameters, updated_control_variate, response_dict

    #orders the connected client to evaluate the given parameters
    def evaluate(self, model_parameters, config_dict):
        self.check_disconnection()
        #convert state_dict inside model_parameters to bytes
        buffer = BytesIO()
        torch.save(model_parameters, buffer)
        buffer.seek(0)
        model_parameters_bytes = buffer.read()
        #convert config_dict to bytes
        config_dict_bytes = json.dumps(config_dict).encode("utf-8")
        #send bytes to client
        eval_order_message = EvalOrder(
            modelParameters = model_parameters_bytes,
            configDict = config_dict_bytes)
        message_to_client = ServerMessage(evalOrder = eval_order_message)
        self.send_buffer.put(message_to_client)
        #get response dict as bytes from client
        client_message = self.recieve_buffer.get()
        eval_response_message = client_message.evalResponse
        response_dict_bytes = eval_response_message.responseDict
        response_dict = json.loads(response_dict_bytes.decode("utf-8"))
        return response_dict

    #orders the client to set its own parameters as the ones passed
    def set_parameters(self, model_parameters):
        self.check_disconnection()
        buffer = BytesIO()
        torch.save(model_parameters, buffer)
        buffer.seek(0)
        model_parameters_bytes = buffer.read()
        set_parameters_order_message = SetParamsOrder(modelParameters = model_parameters_bytes)
        message_to_client = ServerMessage(setParamsOrder = set_parameters_order_message)
        self.send_buffer.put(message_to_client)
        #client sends an empty set params message as response
        self.recieve_buffer.get()

    def check_disconnection(self):
        if not self.is_connected:
            raise Exception(f"Cannot execute command. {self.client_id} is disconnected.")

    def is_disconnected(self):
        return not self.is_connected

    #orders the client to disconnect. if a reconnect is specified (in seconds),
    #the client will attempt to reconnect after that time
    def disconnect(self, reconnect_time = 0, message = "Thank you for participating."):
        self.check_disconnection()
        disconnect_order_message = DisconnectOrder(reconnectTime = reconnect_time, message = message)
        message_to_client = ServerMessage(disconnectOrder = disconnect_order_message)
        self.send_buffer.put(message_to_client)
        