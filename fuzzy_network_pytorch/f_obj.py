import numpy as np 
from sklearn.metrics import mean_squared_error

from fuzzy_network.FuzzyNetwork import FuzzyNetwork

def f_obj(model:FuzzyNetwork, observations:np.array, err_fn='mse') -> float:
  '''
  The objective function. The parameters: all the model's trainable parameters AND all the input values (observations)
  We're gonna use this for calculating the gradients (partial derivatives wrt the trainable paramters)
  '''
  preds = model.inference(observations=observations)
  error = 0
  if err_fn == 'mse':
    error = mean_squared_error(preds, observations)
  if err_fn == 'l2':
    error = np.linalg.norm(preds - observations)
  return error
