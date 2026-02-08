from numpy.linalg import norm
import numpy as np

from FAE.Input import Input 
  

def stopping_crit_reached(k, grad_vec, MAX_ITERATION=8, τ=0.0001):
  '''
    Returns: True, if the stopping criteria is satisfied -> the algorithm can be stopped 

  '''

  if k >= MAX_ITERATION:
    return True


  if np.linalg.norm(grad_vec) < τ:
    return True

  return False

