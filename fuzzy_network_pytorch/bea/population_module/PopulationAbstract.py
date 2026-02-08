
from abc import ABC
import random
from fuzzy_network.bea.Input import InputBEA
from fuzzy_network.bea.bacterium_modul.BacteriumAbstract import BacteriumAbstract
import process_pool
from fuzzy_network.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists
import numpy.random as rnd
from fuzzy_network_pytorch import levenberg_marquardt_pytorch as tlm
import torch 
from torch.nn.utils.stateless import functional_call

def smap(f):
  return f()

class PopulationAbstract(ABC):

  def __init__(self, inp:InputBEA, MyBacteriumConcreteClass:BacteriumAbstract, model_phenotype, loss_fn=tlm.MSELoss()):
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
    def evaluate_population(self, pop_matrix: torch.Tensor) -> torch.Tensor:
        """
        Vectorized evaluation of all individuals & clones without explicit Python loops.
        pop_matrix: [R, n_ind, n_genes] where R = 1 + n_clone
        Returns: losses tensor shaped [R, n_ind]
        NOTE: This assumes the flattened parameter ordering for each individual
              matches the model's named_parameters ordering.
        """
        device = pop_matrix.device
        R, n_ind, n_genes = pop_matrix.shape
        # Flatten to [R * n_ind, n_genes] for batch processing
        batch_matrix = pop_matrix.reshape(R * n_ind, n_genes)

        # Prepare model template (we use the phenotype from first bacterium)
        base_model = self.population[0].model_phenotype
        base_model = base_model.to(device)
        base_params = dict(base_model.named_parameters())
        param_shapes = [p.shape for p in base_params.values()]
        param_sizes = [p.numel() for p in base_params.values()]

        # Split flat parameter vectors into parameter blocks that map to model params
        split_params = torch.split(batch_matrix, param_sizes, dim=1)

        def fwd_single(params_flat, X):
            # rebuild param dict
            param_dict = {}
            idx = 0
            for (name, p), size in zip(base_params.items(), param_sizes):
                block = params_flat[idx: idx + size]
                param_dict[name] = block.view_as(p)
                idx += size
            preds = functional_call(base_model, param_dict, (X,))
            loss = self.loss_fn(preds, Y)  # returns scalar per model
            return loss

        # Prepare inputs as tensors on device
        X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
        Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

        # Use torch.vmap if available (PyTorch >=2.0). If not, fall back to a loop (still on GPU).
        try:
            # batch_matrix: [R*n_ind, n_genes] -> compute scalar loss per row -> [R*n_ind]
            losses_flat = torch.vmap(lambda params: fwd_single(params, X))(batch_matrix)
        except Exception:
            # Fallback (less efficient): loop on GPU
            losses_list = []
            for i in range(batch_matrix.size(0)):
                losses_list.append(fwd_single(batch_matrix[i], X).unsqueeze(0))
            losses_flat = torch.cat(losses_list, dim=0)

        # reshape back to [R, n_ind]
        losses = losses_flat.view(R, n_ind)
        return losses

    def mutation(self):
      """
      Vectorized, GPU-compatible bacterial mutation for fuzzy-network bacteria.
      Each gene = 4-dimensional vector. Random block in [-0.3, 1.3], sorted along last axis.
      """
      import torch
      import torch.nn.functional as F
      from torch.nn.utils.stateless import functional_call

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      n_ind = len(self.population)
      n_clone = getattr(self.inp, "n_clone", 1)
      R = 1 + n_clone  # total rows = original + clones

      # Assume all models have same shape [n_genes, 4]
      n_genes, gene_dim = self.population[0].model.shape

      # Initialize population tensor: [R, n_ind, n_genes, 4]
      pop_matrix = torch.zeros((R, n_ind, n_genes, gene_dim), device=device, dtype=torch.float32)
      for i, b in enumerate(self.population):
          m = b.model
          if not torch.is_tensor(m):
              m = torch.tensor(m, dtype=torch.float32, device=device)
          else:
              m = m.to(device, dtype=torch.float32)
          pop_matrix[0, i] = m
      # Clone originals to the remaining rows
      pop_matrix[1:] = pop_matrix[0].unsqueeze(0).repeat(n_clone, 1, 1, 1)

      # Get gene groups for mutation
      gene_groups = get_rnd_geneId_lists(n_genes)

      # Prepare data tensors for evaluating clones
      X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
      Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)
      loss_fn = self.loss_fn.to(device)
      model_template = self.population[0].model_phenotype.to(device)

      # --- main loop over gene groups ---
      for group_idx, geneIds in enumerate(gene_groups):
          geneIds_t = torch.tensor(geneIds, dtype=torch.long, device=device)
          k = len(geneIds)
          if k == 0:
              continue

          # --- Random mutation for clones ---
          rand_block = (torch.rand((n_clone, n_ind, k, 4), device=device) * 1.6) - 0.3
          rand_block, _ = torch.sort(rand_block, dim=3)
          pop_matrix[1:, :, geneIds_t, :] = rand_block

          # --- Evaluate all candidates (R * n_ind total models) ---
          # Flatten each model to [n_genes*4]
          flat_models = pop_matrix.reshape(R * n_ind, n_genes * 4)

          # Vectorized forward loss computation
          def fwd_loss(flat_params):
              # convert flat params back to model parameter tensor for model_phenotype
              self.model_phenotype.set_trainable_params(flat_params)
              with torch.no_grad():
                  preds = model_template(X)
              return loss_fn(preds, Y)

          try:
              losses = torch.vmap(fwd_loss)(flat_models)
          except Exception:
              losses = torch.stack([fwd_loss(p) for p in flat_models])

          losses = losses.view(R, n_ind)

          # --- Selection: choose best row for each individual ---
          best_rows = torch.argmin(losses, dim=0)

          # Update originals + clones
          for i, b in enumerate(self.population):
              row = best_rows[i].item()
              best_model = pop_matrix[row, i].clone().detach().cpu()
              b.model = best_model
              b._error = float(losses[row, i].item())
              # Overwrite all clones with best model
              pop_matrix[:, i] = best_model.to(device)

      if device.type == "cuda":
          torch.cuda.synchronize()

      return self



  def gene_transfer(self):
    '''
      Performs the Gene Transfer operation on the population.
    '''
    print('Gene transfer')
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
     