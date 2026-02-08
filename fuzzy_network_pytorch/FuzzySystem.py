import copy
import numpy as np

from BMA_FUZZY.fuzzy.mamdani_inference import mamdaniInference_AntesCons

class FuzzySystem():
  def __init__(self, in_dim, out_dim, nr_rules) -> None:
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.nr_rules = nr_rules

    self.Antes = np.zeros((self.nr_rules, self.in_dim, 4))
    self.Cons = np.zeros((self.nr_rules, self.out_dim, 4))

  def __str__(self) -> str:
    msg = f'### Fuzzy System ###\nin_dim: {self.in_dim} \nout_dim: {self.out_dim} \nnr_layers: {self.nr_rules} \n\nAntes: \n{self.Antes} \nCons:\n{self.Cons}\n### END ###\n'
    return msg

  def init_random(self, PADDING_RATE=.3):
    '''
      Random initialization of the fuzzy system.

      Eg:
      >>> f.random_init()
    '''

    for i in range(self.nr_rules):
      for j in range(self.in_dim):
        self.Antes[i][j] = generate_abcd(PADDING_RATE=PADDING_RATE)

    for i in range(self.nr_rules):
      for j in range(self.out_dim):
        self.Cons[i][j] = generate_abcd(PADDING_RATE=PADDING_RATE)

  def genes(self):
    x = self.Antes
    x = x.reshape((self.nr_rules * self.in_dim, 4))
    y = self.Cons
    y = y.reshape((self.nr_rules * self.out_dim, 4))
    x = np.vstack((x, y))
    return x

  def genes_len(self):
    return len(self.genes())

  def set_by_genes(self, genes):
    assert len(genes) == (self.nr_rules * self.in_dim) + (self.nr_rules * self.out_dim), '[-] The length of the genes must matching!'
    len_antes_genes = self.nr_rules * self.in_dim
    len_cons_genes = self.nr_rules * self.out_dim
    
    antes = genes[:len_antes_genes]
    antes = antes.reshape((self.nr_rules, self.in_dim, 4))
    self.Antes = antes

    cons = genes[len_antes_genes:]
    cons = cons.reshape((self.nr_rules, self.out_dim, 4))
    self.Cons = cons

  def params(self):
    params = copy.deepcopy(self.Antes.reshape(-1))
    params = np.concatenate((params, self.Cons.reshape(-1)))
    return params

  def params_len(self):
    return len(self.params())

  def set_by_params(self, params):
    antes_shape = self.Antes.shape
    cons_shape = self.Cons.shape
    antes_len = self.nr_rules * self.in_dim * 4

    self.Antes = copy.deepcopy(params[:antes_len])
    self.Cons = copy.deepcopy(params[antes_len:])
    self.Antes = self.Antes.reshape(antes_shape)
    self.Cons = self.Cons.reshape(cons_shape)

  def inference(self, inputs):
    '''
    Eg:
    -------
    >>> fs.Antes = np.array([[[1,2,2,3],[1,3,3,5]], [[2,4,4,6],[1,2,2,3]]])
    >>> fs.Cons = np.array([[[0,0,0,0], [1,1,1,1], [3,4,4,5]], [[2,2,2,2], [3,3,3,3], [4,5.5,5.5,7]]])
    >>> inputs = np.array([[2.6,2.4], [2.2, 2.3]])
    >>> fs.inference(inputs=inputs)

    >>> # array([[0.        , 0.        , 4.81672598],
    >>> #    [0.        , 0.        , 4.36774194]])
    '''
    return mamdaniInference_AntesCons(Antes=self.Antes, Cons=self.Cons, inputs=inputs)

def generate_abcd(PADDING_RATE=.3) -> tuple:
  """ Generates [abcd] breakpoint values for trapezoidal membership function """
  bound = [0, 1]
  padding = PADDING_RATE * (bound[1] - bound[0])
  nums = np.random.uniform(low=bound[0] - padding, high=bound[1] + padding, size=4)
  nums.sort()
  return nums # a, b, c, d