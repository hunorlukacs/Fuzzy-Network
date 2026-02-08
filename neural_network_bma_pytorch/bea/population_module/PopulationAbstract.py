
from abc import ABC
import random
from neural_network_bma_pytorch.bea.Input import InputBEA
from neural_network_bma_pytorch.bea.bacterium_modul.BacteriumAbstract import BacteriumAbstract
import process_pool
from neural_network_bma_pytorch.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists
import numpy.random as rnd
import torch.nn as nn
import torch
import torch.nn.functional as F
from neural_network_bma_pytorch import levenberg_marquardt_pytorch as tlm
from torch.nn.utils.stateless import functional_call

def smap(f):
  return f()

class PopulationAbstract(ABC):

  def __init__(self, inp:InputBEA, MyBacteriumConcreteClass:BacteriumAbstract, model_phenotype: nn.Module, loss_fn=tlm.MSELoss()):
      super().__init__()
      self.inp = inp
      self.MyBacteriumConcreteClass = MyBacteriumConcreteClass # Lazy initialization
      self._population:list[MyBacteriumConcreteClass] = None
      self.model_phenotype = model_phenotype
      self.loss_fn = loss_fn

  @property
  def population(self):
    ''' population: List[BacteriumAbstract] '''
    if not self._population:
      # Lazy initialization
      self._population = [self.MyBacteriumConcreteClass(inp=self.inp, model_phenotype=self.model_phenotype) for _ in range(self.inp.n_ind) ]
    return self._population

  @population.setter
  def population(self, new_population):
    self._population = new_population

  # --- helper function for evaluating all rows of population matrix ---
  def evaluate_population(self, pop_matrix):
      """
      Vectorized evaluation of all individuals & clones without explicit Python loops.
      pop_matrix: [1 + n_clone, n_ind, n_genes]
      Returns: [1 + n_clone, n_ind] tensor of losses.
      """
      device = pop_matrix.device
      n_total = pop_matrix.shape[0] * pop_matrix.shape[1]  # total models in matrix
      n_genes = pop_matrix.shape[2]

      # Flatten to [n_total, n_genes]
      batch_matrix = pop_matrix.reshape(n_total, n_genes)

      # Prepare model template (any individual's phenotype)
      base_model = self.population[0].model_phenotype
      base_params = dict(base_model.named_parameters())

      # Prepare input & output (same for all clones)
      X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
      Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

      # Map each parameter vector → dict matching model’s parameters
      # (Assumes flattened parameter order matches model.named_parameters())
      param_shapes = [p.shape for p in base_params.values()]
      param_sizes = [p.numel() for p in base_params.values()]
      split_params = torch.split(batch_matrix, param_sizes, dim=1)
      param_dicts = []
      for param_values in split_params:
          param_dicts.append(param_values)

      # --- Vectorized forward pass using vmap + functional_call ---
      def fwd_single(params_flat, X):
          # rebuild param dict
          param_dict = {}
          idx = 0
          for k, p in base_params.items():
              num = p.numel()
              param_dict[k] = params_flat[idx:idx+num].view_as(p)
              idx += num
          preds = functional_call(base_model, param_dict, (X,))
          loss = self.loss_fn(preds, Y)
          return loss

      # Vectorized computation across all parameter sets
      losses = torch.vmap(lambda params: fwd_single(params, X))(batch_matrix)
      return losses.reshape(1 + self.inp.n_clone, len(self.population))

  def mutation(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_ind = len(self.population)
    n_clone = self.inp.n_clone
    n_genes = self.population[0].get_chromosome_length()

    # Create the population matrix [1 + n_clone, n_ind, n_genes]
    # Row 0 = original bacteria, rows 1..n_clone = clones
    pop_matrix = torch.zeros((1 + n_clone, n_ind, n_genes), device=device)
    for i, b in enumerate(self.population):
        pop_matrix[0, i] = b.model
    pop_matrix[1:] = pop_matrix[0].unsqueeze(0).repeat(n_clone, 1, 1)

    X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
    Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

    gene_groups = get_rnd_geneId_lists(n_genes, 4)
    # -------------------------------------------------------------------

    for group_idx, geneIds in enumerate(gene_groups):
        geneIds_t = torch.tensor(geneIds, dtype=torch.long, device=device)

        # ---- BEFORE mutation ----
        # losses_before = self.evaluate_population(pop_matrix)
        # print(f"\n=== BEFORE Mutation Group {group_idx+1}/{len(gene_groups)} ===")
        # for i in range(n_ind):
        #     print(f"Individual {i}, Losses: {losses_before[:, i].cpu().numpy()}")

        # ---- Apply mutation only to clones (keep originals intact before selection) ----
        rand_mut = (torch.rand((n_clone, n_ind, len(geneIds_t)), device=device) *
                    (self.population[0].MAX_WEIGHT - self.population[0].MIN_WEIGHT) +
                    self.population[0].MIN_WEIGHT)
        pop_matrix[1:, :, geneIds_t] = rand_mut

        # ---- AFTER mutation ----
        losses_after = self.evaluate_population(pop_matrix)
        # print(f"\n=== AFTER Mutation Group {group_idx+1}/{len(gene_groups)} ===")
        # for i in range(n_ind):
        #     print(f"Individual {i}, Losses: {losses_after[:, i].cpu().numpy()}")

        # ---- Selection: include original (row 0) in comparison ----
        best_idx = torch.argmin(losses_after, dim=0)  # best row (could be 0 or a clone)

        # Update population and overwrite *both* original + clones
        for i, b in enumerate(self.population):
            best_row = best_idx[i]
            best_model = pop_matrix[best_row, i].clone()

            # Update original individual
            b.model = best_model
            b._error = losses_after[best_row, i].item()

            # Overwrite all rows (original + clones) with best
            pop_matrix[:, i] = best_model

        # ---- Recompute losses for tracking improvement ----
        # losses_post_selection = self.evaluate_population(pop_matrix)
        # print(f"\n=== AFTER SELECTION (Group {group_idx+1}/{len(gene_groups)}) ===")
        # for i in range(n_ind):
        #     print(f"Individual {i}, Losses: {losses_post_selection[:, i].cpu().numpy()}")

    torch.cuda.synchronize()




  def gene_transfer(self):
    '''
      Performs the Gene Transfer operation on the population.
    '''
    n_ind, n_inf, SUBSAMPLING_ENABLED = self.inp.n_ind, self.inp.n_inf, self.inp.SUBSAMPLING_ENABLED

    if n_ind == 1:
      ''' if there is only one individual (eq. test purposes) -> geneTransfer is not needed '''
      return
    len_chromosome = self.population[0].get_chromosome_length()

    ''' Subsampling '''
    if SUBSAMPLING_ENABLED:
      subsampl_ind = generate_rand_indeces(self.inp)
    else:
      subsampl_ind=None

    ### START CODE ###
    for _ in range(n_inf):
      ''' original version '''
      self.population.sort(key=lambda indiv: indiv.error)
      # self.population.sort(key=lambda indiv: indiv.error(indeces=subsampl_ind))
      donorId = rnd.choice(range(n_ind//2))
      acceptorId = rnd.choice(range(n_ind//2, n_ind))

      geneIds = random.sample(range(len_chromosome), random.randint(1, len_chromosome-1))  # selecting random genes
      genes2transfer = self.population[donorId].get_genes(geneIds=geneIds)
      self.population[acceptorId].set_genes(geneIds=geneIds, new_genes=genes2transfer)
    ### END CODE ### 

  def getdata_as_dict(self):
    '''
      Returns all the data wich is relevant to save.

      Returns:
      --------
      data: dict
            Contains: population
    '''
    data = {}
    data['population'] = self.population
    data['inp'] = self.inp
    return data

  def setdata_from_dict(self, data):
    '''
      Sets the population according to tha data.

      Parameter:
      ---------
      data: dict
            Contains: population=
    '''
    self.population = data['population']
    return True

  def get_errors(self):
    ''' 
    Returns:
    --------
        list - Each item represents an individual from the population. An item: [{individual Index}, {individual error}]
    '''
    return [[id, indiv.get_err().round(2)] for id, indiv in enumerate(self.population)]
     