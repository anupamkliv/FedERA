import os
import sys
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federa.server.src.server_lib import save_intial_model
from ..misc import get_config, tester



def create_train_test_for_four_clients():
    """
    Verify the scalability of clients using four clients by implementing the following function
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '4')
            save_intial_model(config['server'])

        def test_four_clients(self):
            print("\n== Testing  for #client=4 ==")
            config = get_config('test_scalability', '4')
            tester(config, 4)

    return TrainerTest


def create_train_test_for_six_clients():
    """
    Verify the scalability of clients using six clients by implementing the following function
    """
     
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '6')
            save_intial_model(config['server'])

        def test_six_clients(self):
            print("\n== Testing  for #client=6 ==")
            config = get_config('test_scalability', '6')
            tester(config, 6)

    return TrainerTest



def create_train_test_for_five_rounds():
    """
    Verify the scalability of CR rounds using five round by implementing the following function
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '5_rounds')
            save_intial_model(config['server'])

        def test_five_communication_rounds(self):
            print("\n== Testing  for Communication Rounds=5 ==")
            config = get_config('test_scalability', '5_rounds')
            tester(config, 2)

    return TrainerTest


def create_train_test_for_ten_rounds():
    """
    Verify the scalability of CR rounds using ten round by implementing the following function
    """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_scalability', '10_rounds')
            save_intial_model(config['server'])

        def test_ten_communication_rounds(self):
            print("\n== Testing  for Communication Rounds=10 ==")
            config = get_config('test_scalability', '10_rounds')
            tester(config, 2)

    return TrainerTest




class TestTrainer_4(create_train_test_for_four_clients()):
    'Test case for four clients'

    
class TestTrainer_6(create_train_test_for_six_clients()):
    'Test case for six clients'

    
class TestTrainer_5_rounds(create_train_test_for_five_rounds()):
    'Test case for five communication rounds'

    
class TestTrainer_10_rounds(create_train_test_for_ten_rounds()):
    'Test case for ten communication rounds'

    

    
if __name__ == '__main__':

    
    unittest.main()
