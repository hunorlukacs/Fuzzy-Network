from definitions import ROOT_DIR

class InputBEA():
  n_gen = 3
  n_ind = 3
  n_clone = 3
  n_inf = 3
  nr_rules = 1
  b_mut=True  # enables bacterial mutation
  b_gt=True   # enables bacterial gene transfer

  observations = None
  desired_outputs = None
  observation_dim = None
  
  SUBSAMPLING_ENABLED = False
  SUBSAMPL_RATIO = 0.3

  MULTIPROCESS_ENABLED = False # True

  SAVE_FILENAME = 'autosave'
  SAVE_PATH = f'{ROOT_DIR}/data/models'

  def input_set_fitData(self):
    ''' Sets the observation_dim attribute '''
    assert self.observations is not None
    # Check if the observations are 1D
    if len(self.observations.shape) == 1:
        self.observation_dim = 1  # Since it's a 1D array, the dimension is 1
    else:
        self.observation_dim = self.observations.shape[1]  # For higher dimensions


def input_set_fitData(inp:InputBEA, observations, desired_outputs):
  ''' Updates the following attributes: `observations`, `desired_outputs`, `observation_dim` '''
  _inp = inp
  _inp.observations = observations
  _inp.desired_outputs = desired_outputs
  _inp.observation_dim = observations.shape[1]
  return _inp
  