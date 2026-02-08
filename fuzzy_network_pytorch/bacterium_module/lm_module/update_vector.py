import numpy as np


def update_vector(grad_vec, gamma):
  '''
  Params:
  -------
    grad_vec: (N, ) gradient vector, where N is the the number
      of the trainable parameters (chromosome length)
  '''
  outer = np.outer(grad_vec, grad_vec)
  I = np.eye(len(grad_vec))
  gamma_I = gamma * I
  parentheses = np.add(outer, gamma_I)
  update_vec = -np.linalg.inv(parentheses) @ grad_vec
  return update_vec