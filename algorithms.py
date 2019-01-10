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
    raise NotImplementedError('Implement me.')

def vanilla_grad_descent(rate, env):
    return grad_descent(lambda pos: -rate * env.gradient(pos), env)



class MomentumStepFunction:
    def __init__(self, alpha, df, rate):
        raise NotImplementedError('Implement me.')

    def __call__(self, pos):
        raise NotImplementedError('Implement me.')


def momentum_grad_descent(rate, env):
    return grad_descent(MomentumStepFunction(0.3, env.gradient, rate), env)



class AdagradStepFunction:
    def __init__(self, df, rate):
        raise NotImplementedError('Implement me.')

        
    def __call__(self, pos):
        raise NotImplementedError('Implement me.')


def adagrad(rate, env):
    return grad_descent(AdagradStepFunction(env.gradient, rate), env)



class RmsPropStepFunction:
    def __init__(self, df, rate, decay_rate, delta=0.000001):
        raise NotImplementedError('Implement me.')
 
    def __call__(self, pos):
        raise NotImplementedError('Implement me.')

def rmsprop(rate, decay_rate, env):
    return grad_descent(RmsPropStepFunction(env.gradient, rate, decay_rate), env)