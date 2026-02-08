import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzy_network_pytorch.bea.Input import InputBEA
from fuzzy_network_pytorch.bea._helper_functions import generate_rand_indeces, get_rnd_geneId_lists
import sys
import torch.nn.functional as F

class Bacterium:
    """
    GPU/vectorized version of Bacterium.

    Internal representation:
      - self.model: torch.Tensor on device, shape (n_genes, 4)
      - self.genotype: same as self.model (alias), kept as torch.Tensor
    External compatibility:
      - phenotype.get_genes() and phenotype.set_trainable_params(...) still works via a fallback to numpy conversions.
      - If the phenotype provides tensor-aware methods (recommended) named
          - set_trainable_params_tensor(tensor)
          - get_genes_tensor() -> tensor
        they will be used to avoid extra CPU-GPU transfers.
    """

    def __init__(self, inp: InputBEA, model_phenotype: nn.Module, device: torch.device = None, verbose=False):
        self.inp = inp
        self.model_phenotype = model_phenotype

        # device selection: prefer inp.device if available, else argument, else cuda if available
        if hasattr(inp, 'device') and isinstance(inp.device, torch.device):
            self.device = inp.device
        elif device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # try to obtain genotype from phenotype (prefer tensor-aware API)
        geno = self.phenotype2genotype(self.model_phenotype)  # this returns numpy or tensor depending on phenotype
        if isinstance(geno, torch.Tensor):
            self.genotype = geno.to(self.device).clone().detach()
        else:
            # assume numpy array
            self.genotype = torch.tensor(geno, dtype=torch.float32, device=self.device)

        # create model: if genotype provided, use it as base; else random init
        if self.genotype.numel() > 0:
            # genotype might be full chromosome (n_genes,4) or flattened
            if self.genotype.ndim == 1:
                # try to reshape if possible
                total = self.genotype.numel()
                if total % 4 != 0:
                    raise ValueError("Genotype length is not divisible by 4")
                n_genes = total // 4
                self.model = self.genotype.view(n_genes, 4).clone().detach()
            else:
                self.model = self.genotype.clone().detach()
        else:
            # fallback to create random model (will call create_model)
            self.model = self.create_model()

        # keep consistent dtype
        self.model = self.model.to(dtype=torch.float32, device=self.device)

        # local loss function (use torch)
        self.loss_fn = nn.MSELoss()

        # cached error (numpy float)
        self._error = float("nan")

    # ------------------
    # Phenotype <-> Genotype helpers (use tensor-aware API if available)
    # ------------------
    def phenotype2genotype(self, phenotype: nn.Module):
        """
        Ask phenotype for genotype. Prefer tensor-returning API if implemented, else call phenotype.get_genes() (numpy).
        """
        # preferred tensor API
        if hasattr(phenotype, 'get_genes_tensor'):
            g = phenotype.get_genes_tensor()
            if isinstance(g, torch.Tensor):
                return g
        # fallback to numpy API
        if hasattr(phenotype, 'get_genes'):
            g = phenotype.get_genes()
            # ensure numpy to torch conversion
            if isinstance(g, np.ndarray):
                return g
            elif isinstance(g, list):
                return np.asarray(g, dtype=np.float32)
            else:
                return g
        # no genotype API -> return empty tensor
        return torch.empty(0, device=self.device)

    def _set_params_to_phenotype(self, flat_tensor: torch.Tensor):
        """
        Push flattened parameters to phenotype.
        If phenotype supports tensor API, call it; otherwise convert to numpy and call set_trainable_params(numpy).
        """
        # ensure detached cpu/numpy only if needed
        if hasattr(self.model_phenotype, 'set_trainable_params_tensor'):
            # directly hand tensor (move to phenotype device if needed)
            try:
                self.model_phenotype.set_trainable_params_tensor(flat_tensor.to(next(self.model_phenotype.parameters()).device))
            except Exception:
                # best-effort fallback to numpy
                arr = flat_tensor.detach().cpu().numpy()
                self.model_phenotype.set_trainable_params(arr.reshape(-1))
        else:
            # fallback: convert to cpu numpy
            arr = flat_tensor.detach().cpu().numpy()
            self.model_phenotype.set_trainable_params(arr.reshape(-1))

    # ------------------
    # Exposed API (keeps behavior similar to original)
    # ------------------
    @property
    def error(self):
        """Lazily compute error if not set."""
        if np.isnan(self._error):
            self._error = self.get_err()
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    def predict(self, Xs: torch.Tensor):
        """
        Predict output for given input tensor.
        This will set phenotype parameters from the internal model (prefer tensor-compatible method).
        """
        flat = self.model.reshape(-1)
        self._set_params_to_phenotype(flat)
        with torch.no_grad():
            # ensure inputs are on same device as phenotype if phenotype expects that
            try:
                Xs_device = Xs.to(next(self.model_phenotype.parameters()).device)
            except Exception:
                Xs_device = Xs.to(self.device)
            outputs = self.model_phenotype(Xs_device)
        # keep result as tensor (moved to self.device)
        return outputs.to(self.device)

    def get_params(self):
        """Return flattened parameters as numpy (compatible with original)."""
        return self.get_genes().reshape(-1)
    
    def get_genes(self, geneIds=None):
        """Return the selected genes as a torch tensor on the same device as the model."""
        # Ensure self.model is a tensor
        if not torch.is_tensor(self.model):
            self.model = torch.tensor(self.model, dtype=torch.float32)

        model_device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")

        if geneIds is None:
            t = self.model.to(model_device)
        else:
            t = self.model[geneIds].to(model_device)

        return t.clone().detach()

    def set_genes(self, geneIds, new_genes):
        """Set specific genes in the model, automatically handling device mismatches."""
        # Ensure both tensors are on the same device
        if not torch.is_tensor(self.model):
            self.model = torch.tensor(self.model, dtype=torch.float32)

        model_device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")

        t = new_genes
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        # Move to same device as model
        t = t.to(model_device)

        # Ensure shape compatibility
        if t.shape[0] != len(geneIds):
            raise ValueError(f"new_genes length mismatch: got {t.shape[0]}, expected {len(geneIds)}")

        self.model[geneIds] = t.clone().detach()
        self.error = float("nan")
        return True


    def create_model(self):
        """
        Creates a random model chromosome with uniform initialization on the device.
        If genotype exists, uses its shape.
        """
        # determine number of genes from genotype if available
        if self.genotype is not None and self.genotype.numel() > 0:
            if self.genotype.ndim == 2 and self.genotype.shape[1] == 4:
                shape = (self.genotype.shape[0], 4)
            elif self.genotype.ndim == 1:
                total = self.genotype.numel()
                assert total % 4 == 0
                shape = (total // 4, 4)
            else:
                shape = (self.genotype.shape[0], 4)
        else:
            # fallback: if Input contains n_rules info, use it; else single gene
            if hasattr(self.inp, 'n_rules'):
                shape = (int(self.inp.n_rules), 4)
            else:
                shape = (1, 4)

        # vectorized uniform in [-0.3, 1.3]
        new_array = (torch.rand(size=shape, device=self.device, dtype=torch.float32) * 1.6) - 0.3
        # sort along last dimension to maintain increasing order per gene
        new_array, _ = torch.sort(new_array, dim=1)
        return new_array.clone().detach()

    def gene_mutation(self, geneIds: list) -> bool:
        """
        Executes mutation on specified gene indices.
        Vectorized: generate new genes for the requested geneIds and assign.
        """
        r, _ = self.model.shape
        assert all(0 <= x < r for x in geneIds), f"[-] Invalid geneIds: {geneIds}"
        k = len(geneIds)
        # generate k x 4 random numbers in [-0.3,1.3]
        new_genes = (torch.rand((k, 4), device=self.device, dtype=torch.float32) * 1.6) - 0.3
        new_genes, _ = torch.sort(new_genes, dim=1)
        self.model[geneIds] = new_genes
        self.error = float("nan")
        return True

    def get_chromosome_length(self):
        """Returns the number of genes in the chromosome."""
        return int(self.model.shape[0])

    def get_err(self):
        """Compute error (loss) between predictions and desired outputs."""
        # set phenotype params from current chromosome
        flat = self.model.reshape(-1)
        self._set_params_to_phenotype(flat)

        with torch.no_grad():
            # move observations to phenotype device if necessary
            try:
                obs_dev = self.inp.observations.to(next(self.model_phenotype.parameters()).device)
            except Exception:
                obs_dev = self.inp.observations.to(self.device)
            preds = self.model_phenotype(obs_dev)

        preds = preds.to(dtype=torch.float32, device=self.device)
        desired = self.inp.desired_outputs.to(dtype=torch.float32, device=self.device)

        # compute MSE (torch)
        loss_val = self.loss_fn(preds, desired).item()
        return float(loss_val)


    def mutation(self, verbose=True):
        """
        Performs Bacterial Mutation (GPU-parallel version) with single-line logging.
        Keeps clones' errors independent for proper statistics.
        """
        device = self.device if hasattr(self, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        len_chromosome = self.get_chromosome_length()
        n_clone = self.inp.n_clone

        # --- Ensure model tensor on GPU ---
        if not torch.is_tensor(self.model):
            base_model = torch.tensor(self.model, device=device, dtype=torch.float32)
        else:
            base_model = self.model.to(device, dtype=torch.float32)

        clones_models = base_model.unsqueeze(0).repeat(n_clone, 1, 1)
        clones_errors = torch.full((n_clone,), float("nan"), device=device)

        batch_inputs = self.inp.observations.to(device)
        desired_outputs = self.inp.desired_outputs.to(device)
        loss_fn = self.loss_fn.to(device)

        gene_groups = list(get_rnd_geneId_lists(len_chromosome))
        n_steps = len(gene_groups)
        init_error = self.error if not torch.isnan(torch.tensor(self.error)) else float("nan")

        for step_idx, geneIds in enumerate(gene_groups, start=1):
            geneIds = torch.tensor(geneIds, device=device, dtype=torch.long)
            k = geneIds.numel()

            # --- Random mutations (vectorized) ---
            if n_clone > 1:
                rand_block = (torch.rand((n_clone - 1, k, 4), device=device) * 1.6) - 0.3
                rand_block, _ = torch.sort(rand_block, dim=2)
                clones_models[1:, geneIds] = rand_block
                clones_errors[1:] = float("nan")

            # --- Evaluate all clones on GPU ---
            preds_all = []
            for i in range(n_clone):
                self.model_phenotype.set_trainable_params(clones_models[i].flatten())
                with torch.no_grad():
                    preds = self.model_phenotype(batch_inputs)
                preds_all.append(preds.unsqueeze(0))

            preds_all = torch.cat(preds_all, dim=0)
            losses = F.mse_loss(preds_all, desired_outputs.unsqueeze(0).expand_as(preds_all), reduction="none")
            clone_loss = losses.mean(dim=(1, 2))
            clones_errors = clone_loss

            # --- Select best clone ---
            best_idx = torch.argmin(clones_errors)
            best_genes = clones_models[best_idx, geneIds].clone()
            clones_models[:, geneIds] = best_genes.unsqueeze(0).expand_as(clones_models[:, geneIds])

            # # --- Single-line logging ---
            # if verbose:
            #     best = clones_errors.min().item()
            #     mean = clones_errors.mean().item()
            #     worst = clones_errors.max().item()
            #     sys.stdout.write(
            #         f"\r[BEA] Step {step_idx:>3}/{n_steps} | Best: {best:.6f} | Mean: {mean:.6f} | Worst: {worst:.6f}"
            #     )
            #     sys.stdout.flush()

        # --- Finalization ---
        best_idx = torch.argmin(clones_errors)
        final_best = clones_errors[best_idx].item()
        self.model = clones_models[best_idx].detach().cpu()
        self.error = final_best

        # if verbose:
        #     if not torch.isnan(torch.tensor(init_error)):
        #         improvement = ((init_error - final_best) / max(init_error, 1e-8)) * 100
        #         sys.stdout.write(
        #             f"\r[BEA] Mutation complete. Final best error: {final_best:.6f} (Improvement: {improvement:.2f}%)\n"
        #         )
        #     else:
        #         sys.stdout.write(f"\r[BEA] Mutation complete. Final best error: {final_best:.6f}\n")
        #     sys.stdout.flush()

        return self
