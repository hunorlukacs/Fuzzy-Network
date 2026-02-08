from fuzzy_network.FuzzyNetwork import FuzzyNetwork
import numpy as np

from fuzzy_network.FuzzySystem import generate_abcd
from fuzzy_network.f_obj import f_obj

def create_model(self):
  model = FuzzyNetwork(layers=self.inp.layers)
  return model


def gene_mutation(self, geneIds:list) -> bool:
  '''
    Executes one mutation on the bacterium's chromosome.
    A gene is a trapeze in this case.
    Genes to be mutated is indicated by the geneIds.

    Example:
    --------
    >>> inp = Input()
    >>> inp.inp_dim=1
    >>> inp.h_dim = 1
    >>> inp.n_rules = 1
    >>> b=Bacterium(inp=inp)
    >>> print(b.get_genes())
    >>> b.gene_mutation([0, -1])
    >>> print(b.get_genes())
    Before mutation:
    >>> [[-0.16638761 -0.1541507   1.19221921  1.27374736]
    >>> [-0.08068711 -0.02492725  0.00771522  1.0266876 ]
    >>> [-0.12424127  0.23529689  0.39714644  1.2768391 ]
    >>> [-0.00166599  0.33973893  0.87523982  1.14810176]]

    After mutation:
    >>> [[-0.21361405  0.13356256  0.67393293  1.22280531]
    >>> [-0.08068711 -0.02492725  0.00771522  1.0266876 ]
    >>> [-0.12424127  0.23529689  0.39714644  1.2768391 ]
    >>> [ 0.25619007  0.69023258  0.97983117  1.25005769]]
  '''

  self.model:FuzzyNetwork
  PADDING_RATE = self.inp.PADDING_RATE
  # if boundaries == None:
  # boundaries = [[0, 1]]*((self._model.obs_dim+1)*self._model.n_rules)

  # new_genes = [generate_abcd(bound=boundaries[geneIdx % c], PADDING_RATE=PADDING_RATE) for geneIdx in geneIds]
  new_genes = [generate_abcd(PADDING_RATE=PADDING_RATE) for _ in geneIds]
  genes = self.model.genes()
  genes[geneIds] = new_genes
  self.model.set_by_genes(genes)
  self.error = np.nan
  return True

def get_chromosome_length(self):
  '''
    Returns the number of genes in the choromosome

    Eg:
    >>> inp = Input()
    >>> inp.inp_dim=3
    >>> inp.h_dim = 2
    >>> inp.n_rules = 2
    >>> b=Bacterium(inp=inp)
    >>> print(b._model.Encoder.shape) # (2, 2, 4, 4)
    >>> print(b._model.Decoder.shape) # (3, 2, 3, 4)
    >>> print(b.get_chromosome_length())
    >>> # 34
  '''
  return self.model.genes_len()


def get_genes(self, geneIds:list=None) -> np.ndarray:
  '''
  For the given geneIds, it returns the corresponding genes
  '''
  if geneIds is not None: 
    return  self.model.genes()[geneIds]
  else:
    return self.model.genes()


def set_genes(self, geneIds:list, new_genes:np.ndarray):
  '''
    Updates it's genes on geneIds localtion with the values of new_genes.

    Note:
    ------
    It updates the error value to NaN!

    Example:
    -------
    
    >>> set_genes(geneIds=[0,5], new_genes=np.array([[0,0,0,0], [0,0,0,0]])

    before the update
    >>> [[[-0.18304855  0.13666382  0.22682816  0.88039112]
    >>>   [-0.25896107 -0.19947514  0.07703904  0.96287034]
    >>>   [ 0.23487434  0.55057925  1.0785298   1.28955343]]
    >>> [[-0.08572487  0.83225531  1.09401513  1.28561429]
    >>>   [-0.01155135  0.54873825  0.70040852  0.99695647]
    >>>   [-0.2877292  -0.0184297   0.05425889  1.0982447 ]]]

    after the update:
    >>> [[[ 0.          0.          0.          0.        ]
    >>>   [-0.25896107 -0.19947514  0.07703904  0.96287034]
    >>>   [ 0.23487434  0.55057925  1.0785298   1.28955343]]
    >>> [[-0.08572487  0.83225531  1.09401513  1.28561429]
    >>>   [-0.01155135  0.54873825  0.70040852  0.99695647]
    >>>   [ 0.          0.          0.          0.        ]]]
  '''

  self._model:FuzzyNetwork
  genes = self.model.genes()
  genes[geneIds] = new_genes
  self.model.set_by_genes(genes)

  self.error = np.nan
  return True

def get_err(self, indeces=None):
  ''' Returns the error (one value) depending on the chosen metric and filtering indeces'''
  if indeces is None:
    return f_obj(model=self.model, observations=self.inp.observations)
  else:
    return f_obj(model=self.model, observations=self.inp.observations[indeces])

