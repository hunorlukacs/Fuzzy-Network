
import numpy as np
import tensorflow as tf


def phenotype2genotype(self, phenotype:tf.keras.Model) -> np.ndarray:
  '''
    Converts the phenotype to genotype.
    The phenotype is a tf.keras.Model object.
    The genotype is a numpy array of shape (, 4).

    Example:
    --------
    >>> phenotype = tf.keras.Model(...)
    >>> genotype = phenotype2genotype(phenotype)
  '''
  return phenotype.get_genes()


def create_model(self):
  shape = self.genotype.shape
  new_array = np.random.uniform(-0.3, 1.3, size=shape) # TODO: check the boundaries
  new_array.sort(axis=1)
  return new_array

def gene_mutation(self, geneIds:list) -> bool:
  '''
    Executes one mutation on the bacterium's chromosome.
    A gene is a trapeze in this case.
    Genes to be mutated is indicated by the geneIds.

  '''

  r, _  = self.model.shape
  assert all(x < r for x in geneIds) and all(x >= 0 for x in geneIds), '[-] Value error: {} ahs to >0 and <{}'.format(geneIds, r)
  
  new_genes = np.random.uniform(-0.3, 1.3, size=(len(geneIds), 4))
  new_genes.sort(axis=1)
  self.model[geneIds] = new_genes
  self.error = np.nan
  return True

def get_chromosome_length(self):
  '''
    Returns the number of genes in the choromosome
  '''
  return self.model.shape[0]


def get_genes(self, geneIds:list) -> np.ndarray:
  '''
  For the given geneIds, it returns the corresponding genes
  '''


  r, _  = self.model.shape
  assert all(x < r for x in geneIds) and all(x >= 0 for x in geneIds), '[-] Value error: {} ahs to >0 and <{}'.format(geneIds, r)

  return self.model[geneIds]


def set_genes(self, geneIds:list, new_genes:np.ndarray):
  '''
    Updates it's genes on geneIds localtion with the values of new_genes.

    Note:
    ------
    It updates the error value to NaN!
  '''

  r, _  = self.model.shape
  assert all(x < r for x in geneIds) and all(x >= 0 for x in geneIds), '[-] Value error: {} ahs to >0 and <{}'.format(geneIds, r)
  
  self.model[geneIds] = new_genes

  self.error = np.nan
  return True

def get_err(self, indeces=None):
  ''' Returns the error (one value) depending on the chosen metric and filtering indeces'''

  self.loss()
  return err(individum=self, inp=self.inp, indeces=indeces, metric=self.inp.METRIC_DISTANCE)
