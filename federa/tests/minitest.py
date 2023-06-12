import unittest
import os
import sys
from .misc import get_config, tester
from ..server.src.server_lib import save_intial_model
import logging

logging.basicConfig(filename='test_results.log', level=logging.INFO)

def create_train_test_for_fedavg():
    """ Verify the FedAvg aggregation algorithm using two clients
    by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('minitest', 'fedavg')
            save_intial_model(config['server'])

        def test_fedavg(self):
            print("\n==Fed Avg==")
            config = get_config('test_algorithms', 'fedavg')
            tester(config, 1)
            logging.info("Test 1 passed\n\tFedAvg algorithm ran successfully\n\tMNIST dataset ran successfully\n\tLeNet model ran succesfully\n")
    return TrainerTest

def create_train_test_for_fedadam():
    """ Verify the FedAdam aggregation algorithm using two clients
    by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('minitest', 'fedadam')
            save_intial_model(config['server'])
        def test_fedadam(self):
            print("\n==Fed Adam==")
            config = get_config('test_algorithms', 'fedadam')
            tester(config, 1)
            logging.info("Test 2 passed\n\tFedAdam algorithm ran successfully\n\tCIFAR100 dataset ran successfully\n\tResnet-18 model ran succesfully\n")
    return TrainerTest

def create_train_test_for_verification_module():
    """
    Verify the verification module using two clients by implementing the following function.
    """

    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('minitest', 'verification')
            save_intial_model(config['server'])

        def test_verification_module(self):
            print('\n==Verfication Module Testing==')
            config = get_config('test_modules', 'verification')
            tester(config, 2)
            logging.info("Test 3 passed\n\tVerification module ran successfully\n\CIFAR10 dataset ran successfully\n\tResnet-50 model ran succesfully\n")
    return TrainerTest


def create_train_test_for_timeout_module():
    """
    Verify the timeout module using two clients by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('minitest', 'timeout')
            save_intial_model(config['server'])

        def test_timeout_module(self):
            print('\n==Timeout Module Testing==')
            config = get_config('test_modules', 'timeout')
            tester(config, 2)
            logging.info("Test 4 passed\n\tTimeout module ran successfully\n\tFashionMNIST dataset ran successfully\n\tLeNet model ran succesfully\n")
    return TrainerTest


def create_train_test_for_intermediate_connection_module():
    """
    Verify the itermeidate connection module using two clients 
    by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('minitest', 'intermediate')
            save_intial_model(config['server'])

        def test_intermediate_module(self):
            print('\n==Intermediate Client Module Testing==')
            config = get_config('test_modules', 'intermediate')
            tester(config, 2, late=True)
            logging.info("Test 5 passed\n\tIntermediate client module ran successfully\n\tCIFAR10 dataset ran successfully\n\tAlexNet model ran succesfully\n")
    return TrainerTest


class TestTrainer_verification(create_train_test_for_verification_module()):
    'Test case for verification module'

    
class TestTrainer_timeout(create_train_test_for_timeout_module()):
    'Test case for timeout module'

    
class TestTrainer_intermediate(create_train_test_for_intermediate_connection_module()):
    'Test case for intermediate client connections module'


class TestTrainer_fedavg(create_train_test_for_fedavg()):
    'Test case for FedAvg'


class TestTrainer_fedadam(create_train_test_for_fedadam()):
    'Test case for FedAdam'

    
if __name__ == '__main__':

    
    unittest.main()