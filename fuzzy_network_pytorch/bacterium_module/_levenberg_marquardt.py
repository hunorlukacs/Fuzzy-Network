from fuzzy_network.FuzzyNetwork import FuzzyNetwork
from fuzzy_network.bacterium_module.lm_module.evaluation import evaluation
from fuzzy_network.bacterium_module.lm_module.generate_rand_indeces import generate_rand_indeces
from fuzzy_network.bacterium_module.lm_module.update_vector import update_vector
from fuzzy_network.bacterium_module.lm_module.bravery_factor import bravery_factor
from fuzzy_network.bacterium_module.lm_module.trust_region import trust_region 
from fuzzy_network.bacterium_module.lm_module.correction import frbs_correction, frbs_correction_sort
from fuzzy_network.bacterium_module.lm_module.stopping_criteria import stopping_crit_reached
from fuzzy_network.f_obj import f_obj

import numpy as np
import numpy.random as rnd
from scipy.optimize import least_squares


def levenberg_marquardt(self):
  '''
    Executes the Levenberg-Marquardt optimization algorithm on the individual.
  '''
  if rnd.rand() > self.inp.lm_prob:
    return self

  gamma_init, SUBSAMPLING_ENABLED, obs_dim = self.inp.gamma_init, self.inp.SUBSAMPLING_ENABLED, self.inp.observation_dim
  b, inp = self.model.params(), self.inp

  k=0
  gamma = gamma_init
  s, J, e = None, None, None
  STOPPING_CRITERIA_SATISFIED = False

  # ### Subsampling ###
  if SUBSAMPLING_ENABLED:
    subsamp_ind = generate_rand_indeces(inp=inp)
    Xs = inp.observations[subsamp_ind]
  else:
    subsamp_ind = None
    Xs = inp.observations
  ### Subsampling ###

  model_updated = FuzzyNetwork(layers=self.model.layers)

  f_obj_old = self.error

  ### START CODE ###
  while not STOPPING_CRITERIA_SATISFIED:

    grad_vec = self.gradient_vector(observations=Xs)
    s = update_vector(grad_vec=grad_vec, gamma=gamma)    

    b_new = frbs_correction_sort(b, s)
    model_updated.set_by_params(b_new)

    f_obj_updated = f_obj(model=model_updated, observations=Xs)
    r = trust_region(grad_vec=grad_vec, f_obj_updated=f_obj_updated, f_obj_old=f_obj_old, update_vec=s)
    gamma = bravery_factor(gamma=gamma, r=r)
    
    b, f_obj_new, isUpdated = evaluation(b_old=b, b_new=b_new, f_obj_old=f_obj_old, f_obj_updated=f_obj_updated)

    if isUpdated:
      f_obj_old = f_obj_updated
      isUpdated = False

    k = k+1

    STOPPING_CRITERIA_SATISFIED = stopping_crit_reached(k=k, grad_vec=grad_vec, MAX_ITERATION=inp.lm_iter, τ=inp.τ)
  ### END CODE ###

  # self.model.set_by_params(b)
  # self.error =  np.nan # f_obj_old
  return b, f_obj_new



### BUILT-IN L-M ###

# Define a function to sort every block of size 4 within the input array
def sort_blocks(arr):
    # Reshape the array to have 4 columns
    reshaped = arr.reshape(-1, 4)
    # Sort each block along the last axis
    reshaped.sort(axis=-1)
    # Return the reshaped array
    return reshaped.reshape(-1)

# Define the model function
def model(x, params, layers):
    params = sort_blocks(params)
    model_tmp = FuzzyNetwork(layers=layers)
    model_tmp.set_by_params(params)
    return model_tmp.inference(observations=x)

# Define the objective function
def objective_function(params, x_data, y_data):
    # Compute model predictions for all data points
    y_pred = model(x_data, params)
    
    # Compute residuals for each data point
    residuals = y_pred - y_data
    
    # Flatten the residuals array to 1-D
    return residuals.flatten()


def levenberg_marquardt_builtin(self):
    # Provide initial guesses for the parameters
    initial_guess = self.model.params()

    # Use least_squares to fit the model to the data
    result = least_squares(objective_function, initial_guess, args=(self.inp.observations, self.inp.observations, layers=self.model.layers))

    # Extract the optimized parameters
    optimized_params = sort_blocks(result.x)