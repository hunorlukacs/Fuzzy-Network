import random
import numpy as np
import numpy.random as rnd
import torch
import warnings

from fuzzy_network_pytorch.Input import InputBEA
from fuzzy_network_pytorch.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists
from torch.nn.utils.stateless import functional_call
from fuzzy_network_pytorch import levenberg_marquardt_pytorch as tlm

def smap(f):
    return f()

class Population:
    def __init__(self, inp: InputBEA, MyBacteriumConcreteClass, model_phenotype, device: torch.device = None, loss_fn=tlm.MSELoss()) -> None:
        """ Initialize the population """
        self.inp = inp
        self.MyBacteriumConcreteClass = MyBacteriumConcreteClass
        self._population: list[MyBacteriumConcreteClass] = None
        self.model_phenotype = model_phenotype
        self.loss_fn = loss_fn
        # device selection: prefer provided device, then inp.device, then cuda if available
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(inp, 'device') and isinstance(inp.device, torch.device):
            self.device = inp.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if CUDA is selected but MULTIPROCESS_ENABLED was requested, disable it and warn
        if torch.cuda.is_available() and getattr(self.inp, "MULTIPROCESS_ENABLED", False):
            warnings.warn("MULTIPROCESS_ENABLED requested but CUDA detected. Disabling multiprocessing to avoid GPU sharing across processes.")
            self.inp.MULTIPROCESS_ENABLED = False

    @property
    def population(self):
        """ List of Bacterium, lazily initialized and placed on the configured device """
        if self._population is None:
            # Lazy initialization; ensure each bacterium uses same device
            pop = [
                self.MyBacteriumConcreteClass(inp=self.inp, model_phenotype=self.model_phenotype, device=self.device)
                for _ in range(self.inp.n_ind)
            ]
            # ensure each individual's model is a tensor on the target device
            for idx, b in enumerate(pop):
                # If b.model is numpy, convert; if tensor, move to device
                if not torch.is_tensor(b.model):
                    b.model = torch.tensor(b.model, dtype=torch.float32, device=self.device)
                else:
                    b.model = b.model.to(dtype=torch.float32, device=self.device)
            self._population = pop
        return self._population

    @population.setter
    def population(self, new_population):
        # Optionally ensure devices consistent for assigned population
        self._population = new_population
        for b in self._population:
            if hasattr(b, 'model') and torch.is_tensor(b.model):
                b.model = b.model.to(self.device)


    # --- helper function for evaluating all rows of population matrix ---
    def evaluate_population(self, pop_matrix: torch.Tensor) -> torch.Tensor:
        """
        Vectorized evaluation of all individuals & clones without explicit Python loops.
        pop_matrix: [R, n_ind, n_genes] where R = 1 + n_clone
        Returns: losses tensor shaped [R, n_ind]
        """
        device = pop_matrix.device
        R, n_ind, n_genes = pop_matrix.shape
        batch_matrix = pop_matrix.reshape(R * n_ind, n_genes)

        base_model = self.population[0].model_phenotype.to(device)
        base_params = dict(base_model.named_parameters())
        param_shapes = [p.shape for p in base_params.values()]
        param_sizes = [p.numel() for p in base_params.values()]
        split_params = torch.split(batch_matrix, param_sizes, dim=1)

        def fwd_single(params_flat, X):
            param_dict = {}
            idx = 0
            for (name, p), size in zip(base_params.items(), param_sizes):
                block = params_flat[idx: idx + size]
                param_dict[name] = block.view_as(p)
                idx += size
            preds = functional_call(base_model, param_dict, (X,))
            loss = self.loss_fn(preds, Y)
            return loss

        X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
        Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

        try:
            losses_flat = torch.vmap(lambda params: fwd_single(params, X))(batch_matrix)
        except Exception:
            losses_list = [fwd_single(batch_matrix[i], X).unsqueeze(0) for i in range(batch_matrix.size(0))]
            losses_flat = torch.cat(losses_list, dim=0)

        losses = losses_flat.view(R, n_ind)
        return losses


    def mutation(self):
        """
        Vectorized, GPU-compatible bacterial mutation for fuzzy-network bacteria.
        Each gene = 4-dimensional vector. Random block in [-0.3, 1.3], sorted along last axis.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_ind = len(self.population)
        n_clone = getattr(self.inp, "n_clone", 1)
        R = 1 + n_clone

        n_genes, gene_dim = self.population[0].model.shape

        pop_matrix = torch.zeros((R, n_ind, n_genes, gene_dim), device=device, dtype=torch.float32)
        for i, b in enumerate(self.population):
            m = b.model
            if not torch.is_tensor(m):
                m = torch.tensor(m, dtype=torch.float32, device=device)
            else:
                m = m.to(device, dtype=torch.float32)
            pop_matrix[0, i] = m

        pop_matrix[1:] = pop_matrix[0].unsqueeze(0).repeat(n_clone, 1, 1, 1)

        gene_groups = get_rnd_geneId_lists(n_genes)

        X = torch.tensor(self.inp.observations, dtype=torch.float32, device=device)
        Y = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=device)

        loss_fn = self.loss_fn.to(device)
        model_template = self.population[0].model_phenotype.to(device)

        for geneIds in gene_groups:
            geneIds_t = torch.tensor(geneIds, dtype=torch.long, device=device)
            k = len(geneIds)
            if k == 0:
                continue

            rand_block = (torch.rand((n_clone, n_ind, k, 4), device=device) * 2) - 1
            rand_block, _ = torch.sort(rand_block, dim=3)
            pop_matrix[1:, :, geneIds_t, :] = rand_block

            flat_models = pop_matrix.reshape(R * n_ind, n_genes * 4)

            def fwd_loss(flat_params):
                self.model_phenotype.set_trainable_params(flat_params)
                with torch.no_grad():
                    preds = model_template(X)
                return loss_fn(preds, Y)

            try:
                losses = torch.vmap(fwd_loss)(flat_models)
            except Exception:
                losses = torch.stack([fwd_loss(p) for p in flat_models])

            losses = losses.view(R, n_ind)

            best_rows = torch.argmin(losses, dim=0)

            for i, b in enumerate(self.population):
                row = best_rows[i].item()
                best_model = pop_matrix[row, i].clone()
                b.model = best_model
                b._error = float(losses[row, i].item())
                pop_matrix[:, i] = best_model

            if device.type == "cuda":
                torch.cuda.synchronize()

        # return self


        
    def gene_transfer(self):
        """ Performs gene transfer in the population (device-safe). """
        n_ind = int(self.inp.n_ind)
        n_inf = int(self.inp.n_inf)
        SUBSAMPLING_ENABLED = getattr(self.inp, "SUBSAMPLING_ENABLED", False)

        if n_ind == 1:
            return

        len_chromosome = int(self.population[0].get_chromosome_length())

        if SUBSAMPLING_ENABLED:
            subsampl_ind = generate_rand_indeces(self.inp)
        else:
            subsampl_ind = None

        for _ in range(n_inf):
            # Sort population in-place by ascending error (handle torch tensors)
            def _err_val(indiv):
                e = indiv.error
                if torch.is_tensor(e):
                    return e.item()
                return float(e)

            self.population.sort(key=_err_val)

            donorId = int(rnd.choice(range(n_ind // 2)))
            acceptorId = int(rnd.choice(range(n_ind // 2, n_ind)))

            # choose random geneIds as Python list
            geneIds = random.sample(range(len_chromosome), random.randint(1, len_chromosome - 1))

            # get_genes returns tensor (device-aware) in updated Bacterium
            genes2transfer = self.population[donorId].get_genes(geneIds=geneIds)
            # ensure genes2transfer is tensor on target device
            if not torch.is_tensor(genes2transfer):
                genes2transfer = torch.tensor(genes2transfer, dtype=torch.float32, device=self.device)
            else:
                genes2transfer = genes2transfer.to(dtype=torch.float32, device=self.device)

            # set_genes will handle device synchronization internally (per updated Bacterium)
            self.population[acceptorId].set_genes(geneIds=geneIds, new_genes=genes2transfer)

        return True

    def getdata_as_dict(self):
        """ Returns all data relevant to save. Note: will include in-memory tensors. """
        data = {
            'population': self.population,
            'inp': self.inp
        }
        return data

    def setdata_from_dict(self, data):
        """ Sets the population from saved data """
        self.population = data['population']
        # ensure models are on correct device
        for b in self.population:
            if hasattr(b, 'model'):
                if not torch.is_tensor(b.model):
                    b.model = torch.tensor(b.model, dtype=torch.float32, device=self.device)
                else:
                    b.model = b.model.to(dtype=torch.float32, device=self.device)
        return True

    def get_errors(self):
        """ Returns a list of [index, error] for each individual (floats on CPU). """
        errors = []
        for idx, indiv in enumerate(self.population):
            # call/get_err (computes on GPU if implemented)
            err = indiv.get_err()
            if torch.is_tensor(err):
                err = err.item()
            errors.append([idx, float(round(err, 6))])
        return errors
