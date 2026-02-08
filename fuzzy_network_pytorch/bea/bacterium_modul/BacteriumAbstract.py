from abc import ABC, abstractmethod
import copy
import math
import numpy as np
from fuzzy_network.bea.Input import InputBEA
from fuzzy_network.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists

class BacteriumAbstract():

  def __init__(self, inp:InputBEA):
    self.inp = inp
    self._error = np.nan

  @property
  def model(self):
    return self._model

  @model.setter
  def model(self, new_model):
    self._model = new_model
    
  @property
  def error(self):
    if not math.isnan(self._error):  # if it's been calculated once already
      return self._error
    else:
      self._error = self.get_err()
      return self._error

  @error.setter
  def error(self, new_error):
    self._error = new_error

  ### START: ABSTRACT METHODS ###

  @abstractmethod
  def create_model(self):
    raise NotImplementedError('[-] Not implemented: create_model()')

  @abstractmethod
  def gene_mutation(self, geneIds:list):
    ''' Executes one mutation on the bacterium's chromosome. '''
    raise NotImplementedError('[-] Not implemented: gene_mutation(geneIds)')

  @abstractmethod
  def get_chromosome_length(self) -> int :
    '''  Returns the number of genes in the choromosome. '''
    raise NotImplementedError('[-] Not implemented: get_chromosome_length()')

  @abstractmethod
  def get_genes(self, geneIds:list) -> np.ndarray:
    ''' For the given geneIds, it returns the corresponding genes. '''
    raise NotImplementedError('[-] Not implemented: get_genes(geneIds)')

  @abstractmethod
  def set_genes(self, geneIds:list, new_genes:np.ndarray):
    ''' Updates it's genes on geneIds localtion with the values of new_genes. '''
    raise NotImplementedError('[-] Not implemented: get_chromosome_length(geneIds, new_genes)')

  @abstractmethod
  def get_err(self, indeces=None) -> float:
    ''' Returns the individum's error. '''
    raise NotImplementedError('[-] Not implemented: get_err(indeces)')

  ### END: ABSTRACT METHODS ###


  def getdata_as_dict(self):
    '''
      Returns all the data wich is relevant to save.

      Returns:
      --------
      data: dict
            Contains: model, error
    '''
    data = {}
    data['model'] = self.model
    data['error'] = self.error
    return data

  def setdata_as_dict(self, data):
    '''
      Set's attributes according to the data.

      Parameter:
      ---------
      data: dict
            Contains: model, error
    '''
    self.model = data['model']
    self.error = data['error']

  def mutation(self):
    ''' 
      Performs the Bacterial Mutation operation on the bacterial individual.   
      Returns: Self
    '''
    print('BacteriumAbstract mutation')
    len_chromosome = self.get_chromosome_length() # The number of genes in the choromosome

    ''' initializing the clones '''
    # clones = [self(self.inp) for _ in range(self.inp.n_clone)]
    clones = [BacteriumAbstract(self.inp) for _ in range(self.inp.n_clone)]
    print('Clones.len', len(clones))
    for c_idx in range(self.inp.n_clone):
      print('before deepcopy')
      clones[c_idx] = copy.deepcopy(self)
      print('after deepcopy')
      
    # ''' Subsampling '''
    # if self.inp.SUBSAMPLING_ENABLED:
    #   subsampl_ind = generate_rand_indeces(self.inp)
    # else:
    #   subsampl_ind=None

    print('before for loop')

    for geneIds in get_rnd_geneId_lists(len_chromosome):
      for c_idx in range(1, self.inp.n_clone):
        print('before gene_mutation')
        clones[c_idx].gene_mutation(geneIds=geneIds)
        print('after gene_mutation')

      ''' selecting the best clone '''
      best_clone = min(clones, key=lambda clone: clone.error)
      print('Best clone error: ', best_clone.error)

      ''' transferring the adecvate genes from the best indiv to the weaker ones '''
      genes_to_transfer = best_clone.get_genes(geneIds=geneIds)
      for c_idx in range(self.inp.n_clone):
        clones[c_idx].set_genes(geneIds=geneIds, new_genes=genes_to_transfer)
        clones[c_idx].error = best_clone.error

    ''' Overwriting the original individual with the best clone '''
    self.model = copy.deepcopy(best_clone.model)
    self.error = best_clone.error
    return self
  

  # def mutation(self):
  #   ''' 
  #     Performs the Bacterial Mutation operation on the bacterial individual.   
  #     Returns: Self
  #   '''
  #   print('BacteriumAbstract mutation')
  #   len_chromosome = self.get_chromosome_length()  # The number of genes in the chromosome

  #   ''' Initializing the clones '''
  #   clones = [BacteriumAbstract(self.inp) for _ in range(self.inp.n_clone)]
  #   print('Clones.len', len(clones))

  #   # Instead of deepcopying the entire object, copy only the model and error
  #   for c_idx in range(self.inp.n_clone):
  #       print('Copying model and error to clone', c_idx)
  #       clones[c_idx].model = copy.deepcopy(self.model)  # Shallow copy of the model
  #       clones[c_idx].error = self.error  # Copy the error

  #   print('before for loop')

  #   for geneIds in get_rnd_geneId_lists(len_chromosome):
  #       for c_idx in range(1, self.inp.n_clone):
  #           print('before gene_mutation')
  #           clones[c_idx].gene_mutation(geneIds=geneIds)
  #           print('after gene_mutation')

  #       ''' Selecting the best clone '''
  #       best_clone = min(clones, key=lambda clone: clone.error)
  #       print('Best clone error: ', best_clone.error)

  #       ''' Transferring the adequate genes from the best individual to the weaker ones '''
  #       genes_to_transfer = best_clone.get_genes(geneIds=geneIds)
  #       for c_idx in range(self.inp.n_clone):
  #           clones[c_idx].set_genes(geneIds=geneIds, new_genes=genes_to_transfer)
  #           clones[c_idx].error = best_clone.error

  #   ''' Overwriting the original individual with the best clone '''
  #   self.model = copy.copy(best_clone.model)  # Shallow copy of the best clone's model
  #   self.error = best_clone.error  # Copy the best clone's error
  #   return self