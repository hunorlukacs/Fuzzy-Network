import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_network_bma_pytorch.bea.Input import InputBEA
from neural_network_bma_pytorch.bea.bacterium_modul.BacteriumAbstract import BacteriumAbstract
from neural_network_bma_pytorch import levenberg_marquardt_pytorch as tlm


class Bacterium(BacteriumAbstract):
    """
    Representation of a bacterium for BEA in PyTorch with optional GPU support.

    Notes:
    - Internal chromosome (self.model) is a torch.Tensor kept on self.device.
    - External API (get_genes, get_params, phenotype2genotype) returns numpy arrays
      to preserve compatibility with existing code.
    """

    def __init__(self, inp: InputBEA, model_phenotype: nn.Module, loss_fn=tlm.MSELoss()) -> None:
        super().__init__(inp, model_phenotype)

        self.MIN_WEIGHT = -2
        self.MAX_WEIGHT = 2

        self.inp = inp
        self.model_phenotype = model_phenotype

        # device: cuda if available else cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # try to move phenotype to device if it supports .to()
        try:
            if hasattr(self.model_phenotype, "to"):
                self.model_phenotype.to(self.device)
        except Exception:
            # ignore if phenotype can't be moved
            pass

        # Genotype: flattened parameters (kept as numpy for compatibility)
        self.genotype = self.phenotype2genotype(self.model_phenotype)

        # Internal model stored as a torch tensor on device
        self.model = self.create_model()  # initialized random genes (torch.Tensor on device)

        self._error = np.nan
        self.loss_fn = loss_fn

    ### Helper methods for robust phenotype param setting ###

    def _torch_to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _set_params_on_phenotype(self, params: torch.Tensor):
        """
        Try to set params on phenotype using a torch.Tensor on device.
        Fall back to numpy array if the phenotype expects numpy.
        """
        try:
            # attempt to pass tensor (preferred)
            self.model_phenotype.set_trainable_params(params)
        except Exception:
            # fallback: pass numpy on CPU
            try:
                self.model_phenotype.set_trainable_params(self._torch_to_numpy(params))
            except Exception:
                # if still failing, raise informative error
                raise RuntimeError(
                    "Failed to set trainable parameters on phenotype. "
                    "Phenotype.set_trainable_params must accept a torch.Tensor or numpy array."
                )

    def _get_params_from_phenotype(self) -> np.ndarray:
        """
        Acquire trainable params from phenotype and return numpy array.
        Accepts either numpy array or torch.Tensor from phenotype.get_trainable_params().
        """
        params = self.model_phenotype.get_trainable_params()
        if isinstance(params, np.ndarray):
            return params
        elif isinstance(params, torch.Tensor):
            return params.detach().cpu().numpy()
        else:
            # attempt to convert
            try:
                return np.asarray(params)
            except Exception:
                raise RuntimeError("phenotype.get_trainable_params() returned unsupported type")

    ### BEA abstract methods ###

    def predict(self, Xs: torch.Tensor):
        """Predict using the phenotype (PyTorch model)."""
        # Make sure inputs are tensors on the device
        if not isinstance(Xs, torch.Tensor):
            Xs = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        else:
            Xs = Xs.to(self.device, dtype=torch.float32)

        # update phenotype params (try tensor on-device, fallback to numpy)
        params_tensor = self.model.flatten()
        # prefer passing a tensor on device to phenotype; if phenotype expects numpy it'll be handled in helper
        try:
            self._set_params_on_phenotype(params_tensor)
        except RuntimeError:
            # let helper raise a more informative error if necessary
            raise

        # run model
        with torch.no_grad():
            # ensure phenotype callable runs on device
            preds = self.model_phenotype(Xs)
        return preds

    def phenotype2genotype(self, phenotype: nn.Module) -> np.ndarray:
        """Convert phenotype to genotype (flattened numpy array)."""
        return self._get_params_from_phenotype()

    def get_params(self):
        """Return flattened params as numpy array."""
        return self.get_genes().reshape(-1)

    def genotype2phenotype(self, genotype: np.ndarray) -> nn.Module:
        """Load genotype into phenotype model and return phenotype."""
        # accept either numpy or torch
        if isinstance(genotype, np.ndarray):
            gen_t = torch.tensor(genotype, dtype=torch.float32, device=self.device)
        elif isinstance(genotype, torch.Tensor):
            gen_t = genotype.to(self.device, dtype=torch.float32)
        else:
            # try to coerce
            gen_t = torch.tensor(np.asarray(genotype), dtype=torch.float32, device=self.device)

        # set params on phenotype (helper will fallback to numpy if needed)
        self._set_params_on_phenotype(gen_t)

        # update internal model tensor too
        self.model = gen_t.clone().detach().reshape(-1)

        return self.model_phenotype

    def create_model(self):
        """Random initialization of genes -> returns torch.Tensor on device."""
        shape = self.genotype.shape  # numpy shape
        # Use torch random uniform directly on device
        # shape might be (N,) or scalar
        torch_shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        rand = (torch.rand(torch_shape, device=self.device, dtype=torch.float32) *
                (self.MAX_WEIGHT - self.MIN_WEIGHT) + self.MIN_WEIGHT)
        return rand.reshape(-1)

    def gene_mutation(self, geneIds: list) -> bool:
        """Mutate selected genes. geneIds can be list/array of indices."""
        if len(geneIds) == 0:
            return False
        # create new genes on device
        new_genes = (torch.rand(len(geneIds), device=self.device, dtype=torch.float32) *
                     (self.MAX_WEIGHT - self.MIN_WEIGHT) + self.MIN_WEIGHT)
        # assign
        self.model[geneIds] = new_genes
        self._error = np.nan
        return True

    def get_chromosome_length(self):
        """Return number of genes in the chromosome."""
        return int(self.model.numel())

    def get_genes(self, geneIds: list = None) -> np.ndarray:
        """Return genes at geneIds or all genes as numpy array (on CPU)."""
        if geneIds is None:
            return self.model.detach().cpu().numpy()
        # allow numpy list/indexing or torch indexing
        if isinstance(geneIds, (list, tuple, np.ndarray)):
            idx = torch.tensor(geneIds, dtype=torch.long, device=self.device)
            return self.model[idx].detach().cpu().numpy()
        else:
            # single index
            return self.model[geneIds].unsqueeze(0).detach().cpu().numpy()

    def set_genes(self, new_genes: np.ndarray, geneIds: list = None):
        """Set genes at geneIds or entire chromosome. Accepts numpy or torch inputs."""
        if isinstance(new_genes, np.ndarray):
            new_t = torch.tensor(new_genes, dtype=torch.float32, device=self.device)
        elif isinstance(new_genes, torch.Tensor):
            new_t = new_genes.to(self.device, dtype=torch.float32)
        else:
            new_t = torch.tensor(np.asarray(new_genes), dtype=torch.float32, device=self.device)

        if geneIds is None:
            self.model = new_t.reshape(-1).clone().detach()
        else:
            # accept list-like indices
            idx = torch.tensor(geneIds, dtype=torch.long, device=self.device)
            self.model[idx] = new_t
        self._error = np.nan
        return True

    def get_err(self):
        """Compute error of this bacterium using the phenotype."""
        # update phenotype params first
        params_tensor = self.model.flatten()
        self._set_params_on_phenotype(params_tensor)

        # prepare inputs and desired outputs on device
        inputs = torch.tensor(self.inp.observations, dtype=torch.float32, device=self.device)
        desired_outputs = torch.tensor(self.inp.desired_outputs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            predictions = self.model_phenotype(inputs)
            # ensure predictions and desired on same device & dtype
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, dtype=torch.float32, device=self.device)
            else:
                predictions = predictions.to(self.device, dtype=torch.float32)

            loss_val = self.loss_fn(predictions, desired_outputs)

        # cache numeric error in numpy-friendly form
        try:
            self._error = float(loss_val.item())
        except Exception:
            # as fallback convert to cpu numpy
            self._error = float(loss_val.detach().cpu().numpy())
        return self._error

    def mutation(self):
        """Perform bacterial mutation (calls parent)."""
        return super().mutation()
