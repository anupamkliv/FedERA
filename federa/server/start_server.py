import argparse

from src.server import server_start
from src.server_lib import save_intial_model

'# the parameters that can be passed while starting the server'
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type= str, default = 'scaffold', help= 'Aggregation algorithm')
parser.add_argument('--clients', type= int, default = 1, help= '#of clients to start')
parser.add_argument('--fraction', type = float, default = 1,
                    help = '''Fraction of clients to select out of the
                    number provided or those available. Float between 0 to 1 inclusive''')
parser.add_argument('--rounds', type= int, default = 1,
                     help = 'Total number of CR')
parser.add_argument('--model_path',  default = 'initial_model.pt',
                     help = "The path of the initial server model's state dict")
parser.add_argument('--epochs', type = int, default = 1,
                     help= '#of epochs for training')
parser.add_argument('--accept_conn',type = int, default = 1,
                    help = '''1, connections accpeted even after FL has begun,
                     else 0.''')
parser.add_argument('--verify', type = int, default = 0,
                     help= '1 for True or 0')
parser.add_argument('--threshold',type = float,default = 0,
                     help = '''Minimum score clients must have in a verification round,
                     .[0,1]''')
parser.add_argument('--timeout', type = int, default=None,
                     help= 'Time limit for training. Specified in seconds')
parser.add_argument('--resize_size', type = int, default = 32, help= 'resize dimension')
parser.add_argument('--batch_size', type = int, default = 32, help= 'batch size')
parser.add_argument('--net', type = str, default = 'LeNet', help= 'client network')
parser.add_argument('--dataset', type = str, default= 'FashionMNIST',
                     help= 'datsset.Use CUSTOME for local dataset')
parser.add_argument('--niid', type = int, default= 1, help= 'value should be [1, 5]')
parser.add_argument('--carbon', type = int, default= 0,
                     help= '1 enable carbon emission at client')
args = parser.parse_args()
                    
                    
configurations = {
    "algorithm": args.algorithm,
    "num_of_clients": args.clients,
    "fraction_of_clients": args.fraction,
    "num_of_rounds": args.rounds,
    "initial_model_path": args.model_path,
    "epochs": args.epochs,
    "accept_conn_after_FL_begin": args.accept_conn,
    "verify": args.verify,
    "verification_threshold": args.threshold,
    "timeout": args.timeout,
    "resize_size": args.resize_size,
    "batch_size": args.batch_size,
    "net": args.net,
    "dataset": args.dataset,
    "niid": args.niid,
    "carbon":args.carbon,
}

                    
'# start the server with the given parameters'
if __name__ == '__main__':
                    
                    
    save_intial_model(configurations)
    server_start(configurations)
