
from BMA_FUZZY_CLASS.helpers.helper_functions.get_boundaries import get_boundaries
from bea.Input import InputBEA
import numpy as np

class Input(InputBEA):
  ''' Config parameters for Bacterial Memetic Algorthm for Fuzzy Rule-Based Autoencoder '''
  
  

  inp_dim = 2
  
  nr_rules = 1    
  layers = [[inp_dim, 1, nr_rules], [1, inp_dim, nr_rules]]
  
  n_gen = 10
  n_ind = 10
  n_clone = 3
  n_inf = 10

  

  THETA = .1
  lm_prob = .2
  lm_iter = 10
  gamma_init = .5
  b_mut=True  # enable bacterial mutation
  b_gt=True   # enable bacterial gene transfer
  b_lm=True   # enable the levenberg marquardt on bacterial evolution
  τ = 0.0001
  
  boundaries = None
  boundaries_consequent = np.array([-0.2, 1.7]) # boundaries for consequent
  observations = None
  observation_dim = None
  desired_outputs = None

  eps = 1e-10

  PADDING_RATE = 0.3
  # METRIC_DISTANCE = 'Canberra'
  METRIC_DISTANCE = 'L2'
  BACT_MUT_DIVIDE_GENELIST_INTO = 3
  SAVE_DIR = 'data/models/'
  SUBSAMPLING_ENABLED = False
  SUBSAMPLING_RATIO = .5
  MULTIPROCESS_ENABLED = False#True

  def input_set_fitData(self):
    assert self.observations is not None
    assert self.desired_outputs is not None
    self.observation_dim = self.observations.shape[1]
    self.boundaries = get_boundaries(obs=self.observations, desired_output=self.desired_outputs)
    if self.boundaries_consequent is not None:
      self.boundaries[-1] = self.boundaries_consequent
  

def input_set_fitData(inp:Input, observations, desired_outputs):
  '''
    Creates an Input object from a Config object.

    Parameter:
    -------
    conf: Config object

    Returns:
    ------
    inp: Input object

    Notes:
    -------
    All the attributes from Config object will be set or added to the Input object.

  '''
  inp_new = Input()
  inp_new = inp

  inp_new.observations = observations
  inp_new.desired_outputs = desired_outputs
  inp_new.observation_dim = observations.shape[1]
  inp_new.boundaries = get_boundaries(obs=observations, desired_output=desired_outputs)
  if inp.boundaries_consequent is not None:
    inp.boundaries[-1] = inp.boundaries_consequent
  
  return inp_new