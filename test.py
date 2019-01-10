import torch
import unittest
from levels import Environment
from torch import tensor
from algorithms import vanilla_grad_descent, momentum_grad_descent
from algorithms import adagrad, rmsprop

class TestLevel(Environment):
    
    def __init__(self):
        Environment.__init__(self, torch.tensor([0.0, 0.0]), 0.1, 100)
    
    def gradient(self, position):
        (x, y) = position
        return torch.tensor([(2*(x+2.5))/5, (2*(y+4))/10])
    
    def _goal_position(self):
        return torch.tensor([-2.5, -4.0])


def check_positions(expected, actual):
    expected = [(round(pos[0].item(), 2), round(pos[1].item(), 2)) for pos in expected]
    actual = [(round(pos[0].item(), 2), round(pos[1].item(), 2)) for pos in actual]
    return expected == actual


class GradientDescentTests(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        pass

    def tearDown(self):
        """Call after every test case."""
        pass

    def test_vanilla(self):
        expected =  [tensor([0., 0.]),
                     tensor([-0.9500, -0.7600]),
                     tensor([-1.5390, -1.3756]),
                     tensor([-1.9042, -1.8742]),
                     tensor([-2.1306, -2.2781]),
                     tensor([-2.2710, -2.6053]),
                     tensor([-2.3580, -2.8703]),
                     tensor([-2.4120, -3.0849]),
                     tensor([-2.4454, -3.2588]),
                     tensor([-2.4662, -3.3996]),
                     tensor([-2.4790, -3.5137]),
                     tensor([-2.4870, -3.6061]),
                     tensor([-2.4919, -3.6809]),
                     tensor([-2.4950, -3.7416]),
                     tensor([-2.4969, -3.7907]),
                     tensor([-2.4981, -3.8304]),
                     tensor([-2.4988, -3.8627]),
                     tensor([-2.4993, -3.8887]),
                     tensor([-2.4995, -3.9099])]
        assert check_positions(expected, 
                               vanilla_grad_descent(0.95, TestLevel())
                               ), "vanilla test failed"
        
    def test_momentum(self):
        expected =  [tensor([0., 0.]),
                     tensor([-0.9500, -0.7600]),
                     tensor([-1.8240, -1.6036]),
                     tensor([-2.3431, -2.3120]),
                     tensor([-2.5584, -2.8452]),
                     tensor([-2.6008, -3.2246]),
                     tensor([-2.5752, -3.4857]),
                     tensor([-2.5390, -3.6618]),
                     tensor([-2.5133, -3.7789]),
                     tensor([-2.5005, -3.8560]),
                     tensor([-2.4965, -3.9065])]
        assert check_positions(expected, 
                               momentum_grad_descent(0.95, TestLevel())
                               ), "momentum test failed"
        
    def test_adagrad(self):
        expected =  [tensor([0., 0.]),
                     tensor([-0.9500, -0.9500]),
                     tensor([-1.4506, -1.5260]),
                     tensor([-1.7698, -1.9453]),
                     tensor([-1.9861, -2.2722]),
                     tensor([-2.1364, -2.5363]),
                     tensor([-2.2421, -2.7541]),
                     tensor([-2.3168, -2.9360]),
                     tensor([-2.3698, -3.0894]),
                     tensor([-2.4074, -3.2194]),
                     tensor([-2.4342, -3.3300]),
                     tensor([-2.4532, -3.4246]),
                     tensor([-2.4667, -3.5054]),
                     tensor([-2.4763, -3.5748]),
                     tensor([-2.4832, -3.6343]),
                     tensor([-2.4880, -3.6854]),
                     tensor([-2.4915, -3.7293]),
                     tensor([-2.4939, -3.7670]),
                     tensor([-2.4957, -3.7995]),
                     tensor([-2.4969, -3.8274]),
                     tensor([-2.4978, -3.8515]),
                     tensor([-2.4985, -3.8721]),
                     tensor([-2.4989, -3.8899]),
                     tensor([-2.4992, -3.9053])]
        assert check_positions(expected, 
                               adagrad(0.95, TestLevel())
                               ), "adagrad test failed"
        
    def test_rmsprop(self):
        expected = [tensor([0., 0.]),
                    tensor([-1.7888, -1.7888]),
                    tensor([-2.2900, -2.6713]),
                    tensor([-2.4413, -3.1919]),
                    tensor([-2.4847, -3.5114]),
                    tensor([-2.4963, -3.7085]),
                    tensor([-2.4992, -3.8288]),
                    tensor([-2.4998, -3.9013])]                
        assert check_positions(expected, 
                               rmsprop(0.4, 0.95, TestLevel())
                               ), "rmsprop test failed"

       

   
if __name__ == "__main__":
    unittest.main() # run all tests