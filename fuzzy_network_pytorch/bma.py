import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import os
from fuzzy_network_pytorch.bea.Input import InputBEA
from fuzzy_network_pytorch.bea.Bacterium import Bacterium
from fuzzy_network_pytorch.bea.bea_optimizer import BEA_optimizer
from fuzzy_network_pytorch import levenberg_marquardt_pytorch as tlm
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

    def get_trainable_params(self):
        """Flatten trainable parameters into 1D tensor."""
        return torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()

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
                loss_fn=nn.MSELoss(),
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
                 bea_enabled=True, grad_based_method_iter=5, bea_verbose=False):

        # --- BEST STATE TRACKER ADDITIONS ---
        self.best_loss = float('inf')
        self.best_state_dict = None
        # ------------------------------------

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
        self.BEA_optimizer.fit(observations=inputs, desired_outputs=targets)
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
            # self.model_wrapper.model.apply_constraints()
            self.optimizer.zero_grad()
            outputs = self.model_wrapper(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            # self.model_wrapper.model.apply_constraints()
        return loss, outputs

    def train_step(self, inputs, targets):
        
        # Initialize _loss and _outputs if it's the first step
        if self._loss is None:
            with torch.no_grad():
                self._outputs = self.model_wrapper(inputs)
                self._loss = self.loss_fn(self._outputs, targets)

        # 1. BEA Step (Keep logic as BEA is derivative-free)
        if self.bea_enabled:
            # BEA finds best params and sets them
            loss_bea, outputs_bea, new_params = self._train_step_bea(inputs, targets)
            
            # Since BEA sets the params, we check if this new loss is better than current
            if loss_bea < self._loss:
                self.model_wrapper.set_trainable_params(new_params) # Accept better BEA params
                self._loss, self._outputs = loss_bea, outputs_bea
            # ELSE: BEA is already designed to keep the best of the population, 
            # so we just keep the previous _loss if BEA didn't improve it.

        # 2. LM Step (No rollback/storage necessary here—let it run)
        if self.lm_enabled:
                self._loss, self._outputs = self._train_step_lm(inputs, targets) 
            

        # 3. Adam Step (No rollback/storage necessary here—let it run)
        if self.adam_enabled:
            # Adam updates params in-place
            self._loss, self._outputs = self._train_step_adam(inputs, targets)

        # --- ALL-TIME BEST STATE CHECK (NEW LOGIC) ---
        if self._loss < self.best_loss:
            self.best_loss = self._loss.item()
            # Safely store the new best model state
            self.best_state_dict = copy.deepcopy(self.model_wrapper.model.state_dict())
        # ---------------------------------------------

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
                # print('loss: ', loss)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            # print('avg_loss: ', avg_loss)
            # print('len(train_loader): ', len(train_loader))
            history["loss"].append(avg_loss)

            if verbose:
                epoch_iter.set_description(f"Epoch {epoch+1}/{epochs}")
                epoch_iter.set_postfix(loss=f"{avg_loss:.6f}")
                epoch_iter.refresh()  # <-- force redraw

        # --- FINAL STEP: Restore the All-Time Best Parameters ---
        if self.best_state_dict is not None:
            self.model_wrapper.model.load_state_dict(self.best_state_dict)
            print(f"\n✅ Restored best model state with loss: {self.best_loss:.6f}")
        # ------------------------------------------------------
        return history

