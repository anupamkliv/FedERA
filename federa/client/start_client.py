from .src.client import client_start
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default = "localhost:8214", help="IP address of the server")
parser.add_argument("--device", type=str, default = "cpu", help="Device to run the client on")
parser.add_argument('--ca', type = str, default= 'ca.pem', help= 'path to CA certificate')
parser.add_argument('--encryption', type = int, default= 0, help= '1 enables ssl encryption')

args = parser.parse_args()

configs = {
    "ip_address": args.ip,
    "wait_time": 10,
    "device": args.device,
    "encryption": args.encryption,
    "ca": args.ca,
}

if __name__ == '__main__':
    client_start(configs)
