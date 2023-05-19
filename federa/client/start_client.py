from .src.client import client_start
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default = "localhost:8214", help="IP address of the server")
parser.add_argument("--device", type=str, default = "cpu", help="Device to run the client on")
args = parser.parse_args()

configs = {
    "ip_address": args.ip,
    "device": args.device
}

if __name__ == '__main__':
    client_start(configs)
