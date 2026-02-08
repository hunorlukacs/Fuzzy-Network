import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

from neural_network_bma_pytorch.bea.Input import InputBEA
from neural_network_bma_pytorch.bea.Bacterium import Bacterium
from neural_network_bma_pytorch.bea.bea_optimizer import BEA_optimizer
from neural_network_bma_pytorch import levenberg_marquardt_pytorch as tlm
# from neural_network_bma_pytorch.torch_levenberg_marquardt import torch_levenberg_marquardt as lm 
# from neural_network_bma_pytorch import lm_TrustRegion as lm_trust_region


# ====================== ModelWrapper ======================

class ModelWrapper(nn.Module):
    """Wraps a PyTorch model for BEA / LM / Adam training."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.trainer = None
        self.loss_fn = None
        self.optimizer = None

    def forward(self, x):
        return self.model(x)

    def get_trainable_params(self, detach=True):
        """Flatten trainable parameters into 1D tensor."""
        if detach:
            return torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()
        else:
            return torch.cat([p.flatten() for p in self.model.parameters()])
    
    def set_trainable_params(self, flat_params):
        """Load flattened parameters into model."""
        flat_params = torch.tensor(flat_params, dtype=torch.float32)
        start = 0
        with torch.no_grad():
            for p in self.model.parameters():
                length = p.numel()
                new_val = flat_params[start:start+length].view(p.shape)
                p.copy_(new_val)
                start += length

    def compile(self,
                loss_fn=tlm.MSELoss(),
                grad_based_optimizer_name='lm',
                learning_rate=1.0,
                n_gen=3, n_ind=3, n_clone=3, n_inf=3,
                b_mut=True, b_gt=True,
                bea_enabled=True,
                grad_based_method_iter=5):

        self.loss_fn = loss_fn
        self.grad_based_optimizer_name = grad_based_optimizer_name
        self.bea_enabled = bea_enabled

        if grad_based_optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        # Initialize trainer
        self.trainer = Trainer(
            model_wrapper=self,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            grad_based_optimizer_name=grad_based_optimizer_name,
            n_gen=n_gen, n_ind=n_ind, n_clone=n_clone, n_inf=n_inf,
            b_mut=b_mut, b_gt=b_gt,
            bea_enabled=bea_enabled,
            grad_based_method_iter=grad_based_method_iter
        )

    def fit(self, train_loader: DataLoader, epochs=1, verbose=1):
        return self.trainer.fit(train_loader, epochs=epochs, verbose=verbose)


# ====================== Trainer ======================

class Trainer:
    """Trainer for BEA / LM / Adam in PyTorch."""

    def __init__(self,
                 model_wrapper: ModelWrapper,
                 loss_fn,
                 optimizer,
                 grad_based_optimizer_name,
                 n_gen=3, n_ind=3, n_clone=3, n_inf=3,
                 b_mut=True, b_gt=True,
                 bea_enabled=True, grad_based_method_iter=5):

        self._loss, self._outputs = None, None
        self.model_wrapper = model_wrapper
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_based_optimizer_name = grad_based_optimizer_name

        # BEA input config
        self.inp = InputBEA()
        self.inp.n_gen, self.inp.n_ind, self.inp.n_clone, self.inp.n_inf = n_gen, n_ind, n_clone, n_inf
        self.b_mut, self.b_gt = b_mut, b_gt
        self.bea_enabled = bea_enabled
        # Initialize BEA statistics
        self.statistics = {'Errors': [], 'Evolution': []}

        if bea_enabled:
            self.BEA_optimizer = BEA_optimizer(
                model_phenotype=self.model_wrapper.model,
                MyBacteriumConcreteClass=Bacterium,
                inp=self.inp
            )

        self.lm_enabled, self.adam_enabled = False, False
        self.lm_iter, self.adam_iter = 0, 0

        if grad_based_optimizer_name == "lm":
            self.lm = tlm.LevenbergMarquardtModule(
                            model=model_wrapper.model,
                            loss_fn=loss_fn,
                            learning_rate=1.0,
                            attempts_per_step=10,
                            solve_method='qr',
                        )
            # self.lm = lm.Trainer(model=self.model_wrapper.model, loss=self.loss_fn, optimizer=self.optimizer)
            self.lm_enabled, self.lm_iter = True, grad_based_method_iter
        elif grad_based_optimizer_name == "lm_trust_region":
            # self.lm = lm_trust_region.Trainer(model=self.model_wrapper.model, loss=self.loss_fn, optimizer=self.optimizer)
            self.lm = tlm.LevenbergMarquardtModule(
                model=model_wrapper.model,
                loss_fn=loss_fn,
                learning_rate=1.0,
                attempts_per_step=10,
                solve_method='qr',
                damping_strategy=tlm.TrustRegionDampingStrategy(),
            )
            self.lm_enabled, self.lm_iter = True, grad_based_method_iter
        elif grad_based_optimizer_name == "adam":
            self.adam_enabled, self.adam_iter = True, grad_based_method_iter

    def _train_step_bea(self, inputs, targets):
        # Disable inner tqdm to avoid printing per generation
        self.BEA_optimizer.fit(observations=inputs, desired_outputs=targets, verbose=False)
        outputs = self.BEA_optimizer.predict(Xs=inputs)
        loss = self.loss_fn(outputs, targets)
        new_params = self.BEA_optimizer.solution.get_params()
        return loss, outputs, new_params

    def _train_step_lm(self, inputs, targets):
        outputs, loss, _, _ = self.lm.training_step(inputs, targets)
        return loss, outputs

    def _train_step_adam(self, inputs, targets):
        outputs, loss = None, None
        for _ in range(self.adam_iter):
            self.optimizer.zero_grad()
            outputs = self.model_wrapper(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
        return loss, outputs

    def train_step(self, inputs, targets):
        # _loss, _outputs = None, None

        if self.bea_enabled:
            params_stored = self.model_wrapper.get_trainable_params()
            loss_bea, outputs_bea, new_params = self._train_step_bea(inputs, targets)
            if self._loss is None or loss_bea < self._loss:
                self.model_wrapper.set_trainable_params(new_params)
                self._loss, self._outputs = loss_bea, outputs_bea
            else:
                self.model_wrapper.set_trainable_params(params_stored)

        if self.lm_enabled:
            # print('loss before LM: ', self._loss)
            params_stored = self.model_wrapper.get_trainable_params()
            for _ in range(self.lm_iter):
                loss_lm, outputs_lm = self._train_step_lm(inputs, targets)
            if self._loss is None or loss_lm < self._loss:
                self._loss, self._outputs = loss_lm, outputs_lm
            # else:
            #     self.model_wrapper.set_trainable_params(params_stored)
            # print('loss after LM: ', self._loss)

        if self.adam_enabled:
            # print('loss before Adam: ', self._loss)
            params_stored = self.model_wrapper.get_trainable_params(detach=False)
            loss_adam, outputs_adam = self._train_step_adam(inputs, targets)
            if self._loss is None or loss_adam < self._loss:
                self._loss, self._outputs = loss_adam, outputs_adam
            # else:
            #     self.model_wrapper.set_trainable_params(params_stored)
            # print('loss after Adam: ', self._loss)

        self.statistics['Errors'].append(self._loss.item())

        return self._loss, self._outputs
        
    def fit(self, train_loader: DataLoader, epochs=1, verbose=1):
        history = {"loss": []}

        # progress bar for epochs
        epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch", dynamic_ncols=True)

        for epoch in epoch_iter:
            epoch_loss = 0.0

            for inputs, targets in train_loader:
                loss, outputs = self.train_step(inputs, targets)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            history["loss"].append(avg_loss)

            if verbose:
                epoch_iter.set_description(f"Epoch {epoch+1}/{epochs}")
                epoch_iter.set_postfix(loss=f"{avg_loss:.6f}")
                epoch_iter.refresh()  # <-- force redraw

        return history

