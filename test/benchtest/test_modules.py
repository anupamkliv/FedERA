import os
import unittest
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federa.server.src.server_lib import save_intial_model
from test.misc import get_config, tester




def create_train_test_for_verification_module():
    """
    Verify the verification module using two clients by implementing the following function.
    """

    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'verification')
            save_intial_model(config['server'])

        def test_verification_module(self):
            print('\n==Verfication Module Testing==')
            config = get_config('test_modules', 'verification')
            tester(config, 2)
    return TrainerTest


def create_train_test_for_timeout_module():
    """
    Verify the timeout module using two clients by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'timeout')
            save_intial_model(config['server'])

        def test_timeout_module(self):
            print('\n==Timeout Module Testing==')
            config = get_config('test_modules', 'timeout')
            tester(config, 2)
    return TrainerTest


def create_train_test_for_intermediate_connection_module():
    """
    Verify the itermeidate connection module using two clients 
    by implementing the following function.
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_modules', 'intermediate')
            save_intial_model(config['server'])

        def test_intermediate_module(self):
            print('\n==Intermediate Client Module Testing==')
            config = get_config('test_modules', 'intermediate')
            tester(config, 2, late=True)
    return TrainerTest


class TestTrainer_verification(create_train_test_for_verification_module()):
    'Test case for verification module'

    
class TestTrainer_timeout(create_train_test_for_timeout_module()):
    'Test case for timeout module'

    
class TestTrainer_intermediate(create_train_test_for_intermediate_connection_module()):
    'Test case for intermediate client connections module'

    
if __name__ == '__main__':

    
    unittest.main()
