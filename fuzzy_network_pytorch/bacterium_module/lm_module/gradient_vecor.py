import numpy as np 
import copy

from fuzzy_network.FuzzyNetwork import FuzzyNetwork
from fuzzy_network.f_obj import f_obj

def gradient_vector(self, observations):
  '''
    The gradient vector, approximated by second-order accurate,
      central finite differences
  '''

  eps = self.inp.eps
  model:FuzzyNetwork = self.model

  params = model.params() 
  grad_vector = np.zeros_like(params)

  model_upper = FuzzyNetwork(layers=self.model.layers) 
  model_lower = FuzzyNetwork(layers=self.model.layers) 

  for p_id in range(len(params)):
    parameters_upper = copy.deepcopy(params)
    parameters_lower = copy.deepcopy(params)
    parameters_upper[p_id] += eps
    parameters_lower[p_id] -= eps

    model_upper.set_by_params(parameters_upper)
    model_lower.set_by_params(parameters_lower)
    err_upper = f_obj(model=model_upper, observations=observations)
    err_lower = f_obj(model=model_lower, observations=observations)
    grad_vector[p_id] = (err_upper - err_lower) / (2*eps)
    
  return grad_vector

# def gradient_vector_general(function, params, observations:np.array, f_obj:function, eps=1e-10):
  '''
    The gradient vector, approximated by second-order accurate, central finite differences
    
    Params:
    ----------
    function: () - (Eg: FuzzyNetwork)
    params: () - function.set_by_params(params)
    f_obj: The objective function
  '''


  params = function.params()
  grad_vector = np.zeros_like(params)

  model_upper = FuzzyNetwork(layers=self.model.layers) 
  model_lower = FuzzyNetwork(layers=self.model.layers) 

  for p_id in range(len(params)):
    parameters_upper = copy.deepcopy(params)
    parameters_lower = copy.deepcopy(params)
    parameters_upper[p_id] += eps
    parameters_lower[p_id] -= eps

    model_upper.set_by_params(parameters_upper)
    model_lower.set_by_params(parameters_lower)
    err_upper = f_obj(model=model_upper, observations=observations)
    err_lower = f_obj(model=model_lower, observations=observations)
    grad_vector[p_id] = (err_upper - err_lower) / (2*eps)
    
  return grad_vector