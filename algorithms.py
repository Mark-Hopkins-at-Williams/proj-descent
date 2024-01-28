import torch
from levels import Environment

def grad_descent(step_fn, env):
    """
    A general-purpose gradient descent algorithm.
    
    step_fn is a function that takes a position (x,y) as input (expressed as
    a 2-dimensional torch.tensor), and returns the relative step to take
    (also expressed as a 2-dimensional torch.tensor).
    
    env is the environment.
    
    The return value should be a list of the positions (including the starting
    positions) visited during the gradient descent. 
    
    """
    while env.status() == Environment.ACTIVELY_SEARCHING:
        # take another step
        raise NotImplementedError('Question 1') 
    #return the list of visited positions
    

def vanilla_grad_descent(rate, env):
    return grad_descent(lambda pos: -rate * env.gradient(pos), env)


class MomentumStepFunction:
    """
    Computes the next step for gradient descent with momentum.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that gradient descent with momentum would take (also expressed as a
    2-dimensional torch.tensor).
        
    """    
    def __init__(self, loss_gradient, learning_rate, momentum_rate):
        raise NotImplementedError('Question 2')
        
    def __call__(self, pos):
        raise NotImplementedError('Question 2')

def momentum_grad_descent(rate, env):
    return grad_descent(MomentumStepFunction(env.gradient, rate, 0.3), env)


class AdagradStepFunction:
    """
    Computes the next step for adagrad.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that adagrad would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, delta = 0.0000001):
        raise NotImplementedError('Question 3')
        
    def __call__(self, pos):
        raise NotImplementedError('Question 3')

def adagrad(rate, env):
    return grad_descent(AdagradStepFunction(env.gradient, rate), env)


class RmsPropStepFunction:
    """
    Computes the next step for RmsProp.

    The __call__ method takes a position (x,y) as its argument (expressed
    as a 2-dimensional torch.tensor), and returns the next relative step
    that RmsProp would take (also expressed as a
    2-dimensional torch.tensor).
        
    """
    def __init__(self, loss_gradient, learning_rate, decay_rate, delta=0.000001):
        raise NotImplementedError('Question 4')
        
    def __call__(self, pos):
        raise NotImplementedError('Question 4')

def rmsprop(rate, decay_rate, env):
    return grad_descent(RmsPropStepFunction(env.gradient, rate, decay_rate), env)