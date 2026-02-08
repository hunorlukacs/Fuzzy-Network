from fuzzy_network.FuzzyNetwork import FuzzyNetwork
from fuzzy_network.Input import Input
from bea.bacterium_modul.BacteriumAbstract import BacteriumAbstract
import numpy as np

class Bacterium(BacteriumAbstract):
  def __init__(self, inp: Input):
    super().__init__(inp)
    self.inp:Input = inp
    self.model:FuzzyNetwork = self.create_model()
    self._error = np.nan

  from ._abstract_functions import create_model , get_chromosome_length, gene_mutation, get_err, set_genes, get_genes
  ### Additional Imported methods ###
  # from ._helper_functions import plot_fuzzyRBS, predict
  from ._levenberg_marquardt import levenberg_marquardt_builtin #levenberg_marquardt
  # from ._exlpicit_formula import explicit_formula_predictions
  from .lm_module.gradient_vecor import gradient_vector
  ### Additional Imported methods  ###

  def levenberg_marquardt_self(self):
    new_params, new_error = self.levenberg_marquardt()
    self.model.set_by_params(new_params)
    self.error = new_error