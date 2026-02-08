from abc import ABC, abstractmethod
import copy
import math
import numpy as np
from neural_network_bma_pytorch.bea.Input import InputBEA
from neural_network_bma_pytorch.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists
import torch
import time
import sys
from tqdm import tqdm
import torch.nn as nn

class BacteriumAbstract():

  def __init__(self, inp:InputBEA, model_phenotype: nn.Module):
    self.inp = inp
    self._error = np.nan
    self.model_phenotype = model_phenotype

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
    """
    GPU-accelerated Bacterial Mutation with batched clone evaluation.
    Includes forced prints that appear even inside tqdm loops.
    """
    print("\n=== Bacterial Mutation Debug Start ===", flush=True)
    start_total = time.time()

    len_chromosome = self.get_chromosome_length()
    device = self.device

    def _p(msg):
        """Force flush printing even under tqdm."""
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    # -----------------------------
    t0 = time.time()
    clones = [copy.deepcopy(self) for _ in range(self.inp.n_clone)]
    for c in clones:
        c.model = c.model.to(device)
        try:
            c.model_phenotype.to(device)
        except Exception:
            pass
    torch.cuda.synchronize(device)
    _p(f"[TIME] Cloning & moving to GPU: {time.time() - t0:.4f}s")

    # -----------------------------
    t1 = time.time()
    if getattr(self.inp, "SUBSAMPLING_ENABLED", False):
        subsampl_ind = generate_rand_indeces(self.inp)
    else:
        subsampl_ind = None
    torch.cuda.synchronize(device)
    _p(f"[TIME] Subsampling setup: {time.time() - t1:.4f}s")

    # -----------------------------
    t2 = time.time()
    n_groups = 0

    for geneIds in get_rnd_geneId_lists(len_chromosome, 4):
        n_groups += 1
        geneIds = torch.tensor(geneIds, dtype=torch.long, device=device)

        # ---- mutation ----
        tm0 = time.time()
        for c_idx in range(1, self.inp.n_clone):
            clones[c_idx].gene_mutation(geneIds=geneIds.tolist())
        torch.cuda.synchronize(device)
        _p(f"[TIME] Group {n_groups}: mutation step: {time.time() - tm0:.4f}s")

        # ---- evaluation ----
        te0 = time.time()
        batch_genes = torch.stack([c.model for c in clones], dim=0)
        inputs = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
        targets = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

        losses = []
        for i in range(batch_genes.shape[0]):
            clones[i]._set_params_on_phenotype(batch_genes[i])
            with torch.no_grad():
                preds = clones[i].model_phenotype(inputs)
                loss = self.loss_fn(preds, targets)
            losses.append(loss)

        loss_vals = torch.tensor([l.item() for l in losses], device=device)
        torch.cuda.synchronize(device)
        _p(f"[TIME] Group {n_groups}: evaluation step: {time.time() - te0:.4f}s")

        # ---- selection & gene transfer ----
        ts0 = time.time()
        best_idx = torch.argmin(loss_vals).item()
        best_clone = clones[best_idx]
        best_genes = best_clone.model[geneIds]
        for c in clones:
            c.model[geneIds] = best_genes
            c._error = loss_vals[best_idx].item()
        torch.cuda.synchronize(device)
        _p(f"[TIME] Group {n_groups}: selection & transfer: {time.time() - ts0:.4f}s")

    torch.cuda.synchronize(device)
    _p(f"[TIME] Total per-group loop: {time.time() - t2:.4f}s")

    # -----------------------------
    t3 = time.time()
    self.model = best_clone.model.clone().detach().to(device)
    self._error = best_clone._error
    try:
        self.model_phenotype.to(device)
    except Exception:
        pass
    torch.cuda.synchronize(device)
    _p(f"[TIME] Final overwrite: {time.time() - t3:.4f}s")

    total_time = time.time() - start_total
    _p(f"=== Total mutation() time: {total_time:.4f}s ===\n")

    return self