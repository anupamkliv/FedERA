import os
import sys
import unittest
# add main directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federa.server.src.server_lib import save_intial_model
from ..misc import get_config, tester




def create_train_test_for_LeNet():
    """ Verify the LeNet-5 CNN model using one client 
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'LeNet')
            save_intial_model(config['server'])

        def test_LeNet(self):
            print("\n==LeNet Testing==")
            config = get_config('test_models', 'LeNet')
            tester(config, 1)
    return TrainerTest


def create_train_test_for_resnet18():
    """ Verify the ResNet18 CNN model using one client 
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet18')
            save_intial_model(config['server'])

        def test_resnet18(self):
            print("\n===Resnet18 Testing==")
            config = get_config('test_models', 'resnet18')
            tester(config, 1)
    return TrainerTest


def create_train_test_for_resnet50():
    """ Verify the ResNet50 CNN model using one client 
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'resnet50')
            save_intial_model(config['server'])

        def test_resnet18(self):
            print("\n==Resnet50 Testing==")
            config = get_config('test_models', 'resnet50')
            tester(config, 1)
    return TrainerTest


def create_train_test_for_vgg16():
    """ Verify the VGG16 CNN model using one client 
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'vgg16')
            save_intial_model(config['server'])

        def test_vgg16(self):
            print("\n==VGG 16 Testing==")
            config = get_config('test_models', 'vgg16')
            tester(config, 1)
    return TrainerTest


def create_train_test_for_AlexNet():
    """ Verify the AlexNet CNN model using one client 
    by implementing the following function """
    
    class TrainerTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            config = get_config('test_models', 'AlexNet')
            save_intial_model(config['server'])

        def test_vgg16(self):
            print("\n==AlexNet Testing==")
            config = get_config('test_models', 'AlexNet')
            tester(config, 1)
    return TrainerTest


class TestTrainer_LeNet(create_train_test_for_LeNet()):
    'Test case for LeNet model'

    
class TestTrainer_resnet18(create_train_test_for_resnet18()):
    'Test case for resnet18 model'

    
class TestTrainer_resnet50(create_train_test_for_resnet50()):
    'Test case for resnet50 model'

    
class TestTrainer_vgg16(create_train_test_for_vgg16()):
    'Test case for vgg16 model'

    
class TestTrainer_AlexNet(create_train_test_for_AlexNet()):
    'Test case for AlexNet model'

if __name__ == '__main__':

    
    unittest.main()
