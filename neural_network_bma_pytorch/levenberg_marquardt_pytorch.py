# Standard library
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, Callable, Generator
import logging
import copy
from typing import Literal, Optional

# PyTorch
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torch.func import functional_call, jacrev, vmap

# PyTorch Lightning & metrics
import pytorch_lightning as pl
from torchmetrics import Metric
import torch
import torch.nn as nn
# Utilities
from tqdm import tqdm

# # Local package imports
# from .damping import DampingStrategy, StandardDampingStrategy
# from .loss import Loss, MSELoss
# from .selection import ParamSelectionStrategy
# from .tree import tree_cat, tree_first_tensor, tree_slice, tree_unsqueeze, tree_indices, tree_to_device
# from .training import TrainingModule




class DampingStrategy(ABC):
    """Base class for damping strategies in Levenberg-Marquardt optimization."""

    @abstractmethod
    def reset(self) -> None:
        """Resets any state to its initial value."""
        pass

    @abstractmethod
    def get_current_damping(self) -> Tensor:
        """Retrieves the current damping factor."""
        pass

    @abstractmethod
    def initialize_step(self, loss: Tensor) -> None:
        """Initializes any state before a training step."""
        pass

    @abstractmethod
    def on_successful_update(self, loss: Tensor) -> None:
        """Adjust the damping factor after a successful update."""
        pass

    @abstractmethod
    def on_unsuccessful_update(self, loss: Tensor) -> None:
        """Adjust the damping factor after an unsuccessful update."""
        pass

    @abstractmethod
    def stop_attempts(self, loss: Tensor) -> bool:
        """Determines if the update should be accepted with no further attempts"""
        pass

    @abstractmethod
    def stop_training(self, loss: Tensor) -> bool:
        """Checks if training should stop based on the damping factor."""
        pass

    @abstractmethod
    def apply(self, JJ: Tensor) -> Tensor:
        """Applies damping to the Gauss-Newton Hessian approximation."""
        pass


class StandardDampingStrategy(DampingStrategy):
    """Standard Levenberg-Marquardt damping strategy.

    This is used inside the Trainer as a generic class. Many damping strategies can be
    implemented using the same interface.
    """

    def __init__(
        self,
        starting_value: float = 1e-3,
        dec_factor: float = 0.1,
        inc_factor: float = 10.0,
        min_value: float = 1e-10,
        max_value: float = 1e10,
        damping_mode: Literal['standard', 'adaptive', 'fletcher'] = 'standard',
        conditional_stopping: bool = True,
        auto_reset: bool = False,
    ) -> None:
        """Initializes `StandardDampingStrategy` instance.

        Args:
            starting_value: Used to initialize the Trainer internal damping_factor.
            dec_factor: Used in the train_step to decrease the damping_factor when
                new_loss < loss.
            inc_factor: Used in the train_step to increase the damping_factor when
                new_loss >= loss.
            min_value: Used as a lower bound for the damping_factor. Higher values
                improve numerical stability in the resolution of the linear system, at
                the cost of slower convergence.
            max_value: Used as an upper bound for the damping_factor, and as a condition
                to stop the training process.
            damping_mode: Specifies the damping mode. Options are:
                - 'standard': Standard damping using the identity matrix (default).
                - 'adaptive': Apply adaptive scaling with max(diagonal(JJ)).
                - 'fletcher': Use Fletcher's modification for damping.
            conditional_stopping: If True, stops training based on damping conditions.
            auto_reset: If True, resets the damping factor when `stop_attempts` is True.
        """
        self.starting_value = torch.tensor(starting_value)
        self.dec_factor = torch.tensor(dec_factor)
        self.inc_factor = torch.tensor(inc_factor)
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)
        self.damping_mode = damping_mode
        self.conditional_stopping = conditional_stopping
        self.auto_reset = auto_reset

        self.damping_factor = torch.tensor(starting_value)

    def reset(self) -> None:
        """Resets the damping factor to the starting value."""
        self.damping_factor = self.starting_value

    def get_current_damping(self) -> Tensor:
        """Retrieves the current damping factor."""
        return self.damping_factor

    def initialize_step(self, loss: Tensor) -> None:
        """Initializes any state before a training step."""
        pass

    def on_successful_update(self, loss: Tensor) -> None:
        """Decreases the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            The decreased damping factor.
        """
        self.damping_factor = torch.max(
            self.damping_factor * self.dec_factor, self.min_value
        )

    def on_unsuccessful_update(self, loss: Tensor) -> None:
        """Increases the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            The increased damping factor.
        """
        self.damping_factor = torch.min(
            self.damping_factor * self.inc_factor, self.max_value
        )

    def stop_attempts(self, loss: Tensor) -> bool:
        """Determines if further attempts should be stopped and performs auto-reset."""
        should_stop = bool((self.damping_factor >= self.max_value).item())
        if self.auto_reset and should_stop:
            self.reset()
        return should_stop

    def stop_training(self, loss: Tensor) -> bool:
        """Determines whether to stop training based on the damping factor.

        Args:
            loss: The current loss value.

        Returns:
            True if the damping factor exceeds the maximum value, False otherwise.
        """
        return self.stop_attempts(loss) and self.conditional_stopping

    def apply(self, JJ: Tensor) -> Tensor:
        """Applies the damping to the Gauss-Newton Hessian approximation.

        Args:
            JJ: The Gauss-Newton Hessian approximation matrix.

        Returns:
            The damped Hessian matrix.
        """
        if self.damping_mode == 'standard':
            damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)
        elif self.damping_mode == 'adaptive':
            damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)
            damping_matrix = damping_matrix * torch.max(torch.abs(torch.diagonal(JJ)))
        elif self.damping_mode == 'fletcher':
            damping_matrix = torch.diag(torch.diagonal(JJ))
        else:
            raise ValueError(
                f"Invalid damping_mode '{self.damping_mode}'. Expected one of "
                f"'standard', 'adaptive', or 'fletcher'."
            )

        return JJ + self.damping_factor * damping_matrix


class TrustRegionDampingStrategy(DampingStrategy):
    """Trust region based Levenberg–Marquardt damping strategy.
    
    This strategy calculates the damping factor (lambda/gamma) updates by comparing 
    the actual reduction in loss to the predicted reduction from the linear model 
    (the gain ratio rho).
    """

    def __init__(
        self,
        starting_value: float = 1.0,
        min_value: float = 1e-10,
        max_value: float = 1e10,
        dec_factor: float = 0.5, 
        inc_factor: float = 4.0, 
        conditional_stopping: bool = True,
        auto_reset: bool = False
    ) -> None:
        """Initializes TrustRegionDampingStrategy.

        Args:
            starting_value: Initial damping factor (gamma).
            min_value: Lower bound for damping factor.
            max_value: Upper bound for damping factor.
            dec_factor: Factor to decrease damping (default 0.5 implies /2).
            inc_factor: Factor to increase damping (default 4.0 implies *4).
            conditional_stopping: If True, stop training if max_value is reached.
            auto_reset: If True, reset gamma when max_value is reached.
        """
        self.starting_value = torch.tensor(starting_value)
        self.min_value = torch.tensor(min_value)
        self.max_value = torch.tensor(max_value)
        self.dec_factor = torch.tensor(dec_factor)
        self.inc_factor = torch.tensor(inc_factor)
        
        self.conditional_stopping = conditional_stopping
        self.auto_reset = auto_reset

        self.damping_factor = torch.tensor(starting_value)
        
        # State storage for Trust Region calculation
        self.E_old: Optional[Tensor] = None
        self.J: Optional[Tensor] = None
        self.residuals: Optional[Tensor] = None
        self.update_vector: Optional[Tensor] = None

    def reset(self) -> None:
        """Resets the damping factor to the starting value."""
        self.damping_factor = self.starting_value.clone()
        self._clear_step_data()

    def get_current_damping(self) -> Tensor:
        """Retrieves the current damping factor."""
        return self.damping_factor
    
    def set_step_data(self, J: Tensor, residuals: Tensor, update_vector: Tensor) -> None:
        """Registers the linear algebra components required for the gain ratio calculation.
        
        This must be called inside the training loop after the update vector is solved 
        but before `on_successful_update` is triggered.

        Args:
            J: The Jacobian matrix.
            residuals: The residual vector.
            update_vector: The calculated step (h or s_k).
        """
        self.J = J
        self.residuals = residuals
        self.update_vector = update_vector

    def _clear_step_data(self):
        self.J = None
        self.residuals = None
        self.update_vector = None
        self.E_old = None

    def initialize_step(self, loss: Tensor) -> None:
        """Initializes state before a training step.
        
        Args:
            loss: The current loss (E_old), equivalent to ||e_k||^2.
        """
        self.E_old = loss

    def _calculate_gain_ratio(self, E_new: Tensor) -> Optional[float]:
        """Calculates rho = (Actual Reduction) / (Predicted Reduction).
        
        Returns:
            float: The gain ratio rho (r_k).
            None: If necessary data (J, residuals) was not provided via `set_step_data`.
        """
        if self.J is None or self.residuals is None or self.update_vector is None or self.E_old is None:
            return None

        with torch.no_grad():
            # Predicted error vec: J * s_k + e_k
            # Note: This assumes the update is additive: b_new = b_old + s_k
            pred_error_vec = torch.matmul(self.J, self.update_vector) + self.residuals
            pred_error_norm = torch.norm(pred_error_vec)

            # Predicted Reduction: ||e_k|| - ||e_k + J*s_k||
            denominator = self.E_old - pred_error_norm
            
            # Actual Reduction: ||e_k|| - ||e_{k+1}||
            numerator = self.E_old - E_new
            
            if torch.abs(denominator) < 1e-8:
                return 0.0
            
            rho = numerator / denominator
            return rho.item()

    def on_successful_update(self, loss: Tensor) -> None:
        """Adjusts damping after a successful step (loss decreased).
        
        Args:
            loss: The new loss value (E_new).
        """
        rho = self._calculate_gain_ratio(loss)

        if rho is None:
            # Fallback: Standard heuristic if Trust Region data is missing
            self.damping_factor = torch.max(
                self.damping_factor * self.dec_factor, self.min_value
            )
            return

        # Trust Region Update Rules (Eq. bravey_fact):
        # if r_k > 0.75: gamma = gamma / 2
        if rho > 0.75:
            self.damping_factor = torch.max(
                self.damping_factor / 2.0, self.min_value
            )
        # if r_k < 0.25: gamma = 4 * gamma
        elif rho < 0.25:
            self.damping_factor = torch.min(
                self.damping_factor * 4.0, self.max_value
            )
        
        # otherwise (0.25 <= r_k <= 0.75): keep gamma same

    def on_unsuccessful_update(self, loss: Tensor) -> None:
        """Adjusts damping after an unsuccessful step (loss increased).
        
        Args:
            loss: The new loss value (E_new).
        """
        # If the step failed (loss increased), r_k is typically negative (Actual Reduction < 0).
        # This satisfies the condition r_k < 0.25, so we multiply by 4.
        
        self.damping_factor = torch.min(
            self.damping_factor * 4.0, self.max_value
        )

    def stop_attempts(self, loss: Tensor) -> bool:
        """Determines if further attempts should be stopped."""
        should_stop = bool((self.damping_factor >= self.max_value).item())
        if self.auto_reset and should_stop:
            self.reset()
        return should_stop

    def stop_training(self, loss: Tensor) -> bool:
        """Checks if training should stop based on the damping factor."""
        return self.stop_attempts(loss) and self.conditional_stopping

    def apply(self, JJ: Tensor) -> Tensor:
        """Applies damping to the approximate Hessian (J^T * J)."""
        # Create identity matrix matching the device and dtype of JJ
        damping_matrix = torch.eye(JJ.shape[0], dtype=JJ.dtype, device=JJ.device)
        
        return JJ + self.damping_factor * damping_matrix
    


class Loss(torch.nn.Module, ABC):
    """Base class for all loss functions using ABC."""

    @abstractmethod
    def forward(self, y_pred: Any, y_true: Any) -> Tensor:
        """Computes the loss between `y_pred` and `y_true`."""
        pass

    @abstractmethod
    def residuals(self, y_pred: Any, y_true: Any) -> Tensor:
        """Computes the residuals between `y_pred` and `y_true`."""
        pass


class MSELoss(Loss):
    """Mean Squared Error loss for regression problems.

    Provides methods to compute the loss and residuals for mean squared error.
    """

    def __init__(self) -> None:
        """Initializes the MeanSquaredError loss function."""
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the mean squared error loss.

        Args:
            y_pred: Predicted tensor.
            y_true: Ground truth target tensor.

        Returns:
            A scalar tensor representing the loss.
        """
        return (y_pred - y_true).square().mean()

    def residuals(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the residuals for mean squared error.

        Args:
            y_pred: Predicted tensor.
            y_true: Ground truth target tensor.

        Returns:
            A tensor representing the residuals.
        """
        return y_pred - y_true


class LossWrapper(Loss):
    """Wrapper for a PyTorch loss function to adapt it to the `Loss` interface.

    This class allows wrapping any existing PyTorch loss function to make it compatible
    with the `Loss` interface, providing methods to compute both the loss and residuals.

    The residuals are computed using the square root trick, where the square root of the
    unreduced loss (`reduction='none'`) is used to derive the per-sample residuals.
    This enables compatibility with the Gauss-Newton framework, allowing diverse loss
    functions (e.g., Cross-Entropy, Huber) to be used in least-squares optimization.
    """

    def __init__(self, loss_fn) -> None:
        """Initializes the LossWrapper with a PyTorch loss function.

        Args:
            loss_fn: A callable PyTorch loss function that accepts arguments in the
                format `(input: Tensor, target: Tensor, reduction: str) -> Tensor`.
        """
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the wrapped loss function.

        Args:
            y_pred: Predicted tensor.
            y_true: Ground truth target tensor.

        Returns:
            Tensor: A scalar tensor representing the loss computed by the wrapped loss.
        """
        return self.loss_fn(y_pred, y_true, reduction='mean')

    def residuals(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Computes the residuals using the wrapped loss function.

        Args:
            y_pred: Predicted tensor.
            y_true: Ground truth target tensor.

        Returns:
            Tensor: A tensor representing the residuals, computed as the square root of
            the element-wise loss values without reduction.
        """
        return torch.sqrt(self.loss_fn(y_pred, y_true, reduction='none'))


class L1Loss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.l1_loss)


class HuberLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.huber_loss)


class CrossEntropyLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.cross_entropy)


class BCELoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.binary_cross_entropy)


class BCEWithLogitsLoss(LossWrapper):
    def __init__(self) -> None:
        super().__init__(torch.nn.functional.binary_cross_entropy_with_logits)


class ParamSelectionStrategy(ABC):
    """Base class for parameter selection strategies.

    Parameter selection strategies determine which subset of model parameters to update
    during a training step. This could reduce computational, memory requirements and
    act as a regularization for large models.
    """

    @abstractmethod
    def select_parameters(self) -> Tensor:
        """Selects a subset of parameters for the current training step.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        pass


class RandomSelectionStrategy(ParamSelectionStrategy):
    """Randomly selects a fixed-size subset of parameters at each step.

    This strategy picks a random subset of the model's parameters from the
    flattened parameter vector. It is often used to limit computations and
    memory usage, making it more feasible to work with very large models.

    Args:
        params: An iterable of model parameters.
        subset_size: The number of parameters to select at each step.

    Raises:
        ValueError: If `subset_size` exceeds the total number of parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        subset_size: int,
    ) -> None:
        self.params = list(params)
        self.total_num_params = sum(p.numel() for p in self.params)

        if subset_size > self.total_num_params:
            raise ValueError(
                f'subset_size ({subset_size}) cannot exceed the total number of '
                f'parameters ({self.total_num_params}).'
            )

        self.subset_size = subset_size
        self.device = self.params[0].device if self.params else torch.device('cpu')

    def select_parameters(self) -> Tensor:
        """Selects a random subset of parameters.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        param_indices = torch.randperm(self.total_num_params, device=self.device)
        selected = param_indices[: self.subset_size]
        return selected.sort().values


class LayerSelectionStrategy(ParamSelectionStrategy):
    """Selects parameters corresponding to a single parameter group each step.

    This strategy returns a contiguous block of parameter indices for exactly one
    parameter tensor at a time (e.g., corresponding to a single layer). The parameter
    selection can be performed randomly or in a round-robin fashion (cyclic mode).

    Args:
        params: An iterable of model parameters.
        mode: The selection mode. Should be either 'random' or 'cyclic'.

    Raises:
        ValueError: If `mode` is not 'random' or 'cyclic'.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        mode: Literal['random', 'cyclic'] = 'random',
    ) -> None:
        if mode not in ('random', 'cyclic'):
            raise ValueError("mode must be either 'random' or 'cyclic'.")

        self.params = list(params)
        self.mode = mode
        self.device = self.params[0].device if self.params else torch.device('cpu')

        # Precompute the slices corresponding to each parameter
        current_index = 0
        self.param_slices: list[tuple[int, int]] = []
        for p in self.params:
            size = p.numel()
            self.param_slices.append((current_index, current_index + size))
            current_index += size

        # For 'cyclic' mode, keep track of the current parameter index
        self.current_param_idx = 0

    def select_parameters(self) -> Tensor:
        """Selects parameters corresponding to a single parameter tensor.

        Returns:
            The indices of the selected parameters in a flattened parameter vector.
        """
        if not self.params:
            return torch.tensor([], dtype=torch.long, device=self.device)

        if self.mode == 'random':
            param_idx = int(
                torch.randint(low=0, high=len(self.params), size=(1,)).item()
            )
        else:
            param_idx = self.current_param_idx
            self.current_param_idx = (self.current_param_idx + 1) % len(self.params)

        start, end = self.param_slices[param_idx]
        return torch.arange(start, end, device=self.device, dtype=torch.long)


logger = logging.getLogger(__name__)


class TrainingModule(ABC):
    """Abstract base class defining a training interface."""

    @abstractmethod
    def training_step(
        self,
        inputs: Any,
        targets: Any,
    ) -> tuple[Any, Tensor, bool, dict[str, Any]]:
        """Performs a single training step."""
        pass

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Returns the model being trained."""
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Returns the device of the model's parameters."""
        pass


class LevenbergMarquardtModule(TrainingModule):
    """Levenberg-Marquardt training module."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Loss | None = None,
        damping_strategy: DampingStrategy | None = None,
        learning_rate: float = 1.0,
        attempts_per_step: int = 10,
        solve_method: Literal['qr', 'cholesky', 'solve'] = 'qr',
        param_selection_strategy: ParamSelectionStrategy | None = None,
        use_vmap: bool = True,
        max_batch_size: int | None = None,
    ) -> None:
        """Initializes `LevenbergMarquardtModule` instance.

        Args:
            model: The model to be trained, expected to inherit from `torch.nn.Module`.
            loss_fn: A custom loss function inheriting from `Loss`.
                Defaults to `MSELoss()`.
            damping_strategy: Damping strategy to use during training.
                Defaults to `StandardDampingStrategy`.
            learning_rate: Specifies the step size for updating the model parameters.
                The update is performed using the formula `w = w - lr * updates`,
                where `updates` are calculated by the Levenberg-Marquardt algorithm.
            attempts_per_step: Defines the maximum number of attempts allowed during a
                training step to compute a valid model update that reduces the loss on
                the current batch. During each attempt, new model parameters are
                computed, and the resulting loss (`new_loss`) is compared to the
                previous loss. If `new_loss < loss`, the new parameters are accepted.
                Otherwise, the old parameters are restored, and a new attempt is made
                with an adjusted damping factor. If the maximum number of attempts is
                reached without reducing the loss, the step is finalized with the last
                computed parameters, even if they do not decrease the loss.
            solve_method: Solver to use for the linear system. Options:
                - 'qr': QR decomposition (robust, slower).
                - 'cholesky': Cholesky decomposition (fast, less stable).
                - 'solve': Direct solve (balanced speed and robustness).
            param_selection_strategy: A `ParamSelectionStrategy` instance defining how
                subsets of parameters are chosen each training_step. If None, all
                parameters are used.
            use_vmap: Specifies whether to use `torch.vmap` for Jacobian computation.
                Enabling `vmap` is generally the preferred choice as it is faster
                and requires less memory, especially for medium to large models.
                For very small models or simple cases, computing the Jacobian
                without `vmap` might be marginally more efficient. Defaults to `True`.
            max_batch_size: If set, the input batch is divided into smaller sub-batches
                of this size when computing the Jacobian and forming the Gauss-Newton
                approximations. Each sub-batch processes a portion of the input data at
                a time, allowing the Hessian approximation and the RHS vector to be
                constructed incrementally. This reduces peak memory usage but can limit
                parallelism, as the computations are partially serialized rather than
                fully utilizing hardware resources for parallel computation.
        """
        self._model = model

        # Set up loss function and damping strategy
        self.loss_fn = loss_fn or MSELoss()
        self.damping_strategy = damping_strategy or StandardDampingStrategy()

        self.learning_rate = learning_rate
        self.attempts_per_step = attempts_per_step
        self.solve_method = solve_method
        self.param_selection_strategy = param_selection_strategy
        self.use_vmap = use_vmap
        self.max_batch_size = max_batch_size

        # Extract trainable parameters
        self._params = {
            n: p for n, p in self._model.named_parameters() if p.requires_grad
        }
        self._num_params = sum(p.numel() for p in self._params.values())

        # Precompute splits for flat_params
        self._splits = [p.numel() for p in self._params.values()]

        # Flatten all trainable parameters into a single tensor
        self.flat_params = torch.cat(
            [p.detach().flatten() for p in self._params.values()]
        )

        # Bind model parameters to slices of the flat parameter tensor
        start = 0
        for _, p in self._params.items():
            size = p.numel()
            p.data = self.flat_params[start : start + size].view_as(p)
            start += size

        # Backup storage for parameters
        self._flat_params_backup: Tensor
        self.backup_parameters()  # Initialize backup with the current parameters

        # Combine named parameters and buffers into a single dictionary for inference
        self._params_and_buffers = {
            **dict(self._model.named_parameters()),
            **dict(self._model.named_buffers()),
        }

        self._batch_size: int | None = None
        self._num_residuals: int | None = None

    @torch.no_grad()
    def backup_parameters(self) -> None:
        """Backs up the current model parameters into a separate tensor."""
        self._flat_params_backup = self.flat_params.clone()

    @torch.no_grad()
    def restore_parameters(self) -> None:
        """Restores model parameters from the backup tensor."""
        self.flat_params.copy_(self._flat_params_backup)

    @torch.no_grad()
    def reset(self) -> None:
        """Resets internal state, including the damping factor and outputs."""
        self._batch_size = None
        self._num_residuals = None
        self.damping_strategy.reset()

    def forward(self, inputs: Any) -> Any:
        """Performs a forward pass using the current model parameters."""
        return functional_call(self._model, self._params_and_buffers, inputs)

    @torch.no_grad()
    def _solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        """Solves the linear system using the specified solver.

        Args:
            matrix: The matrix representing the linear system.
            rhs: The right-hand side vector.

        Returns:
            The solution vector.
        """

        if self.solve_method == 'qr':
            q, r = torch.linalg.qr(matrix)
            y = torch.matmul(q.transpose(-2, -1), rhs)
            return torch.linalg.solve_triangular(r, y, upper=True)
        elif self.solve_method == 'cholesky':
            L = torch.linalg.cholesky(matrix)
            y = torch.linalg.solve_triangular(L, rhs, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif self.solve_method == 'solve':
            return torch.linalg.solve(matrix, rhs)
        else:
            raise ValueError(
                f"Invalid solve_method '{self.solve_method}'. "
                "Choose from 'qr', 'cholesky', 'solve'."
            )

    @torch.no_grad()
    def _apply_updates(self, updates: Tensor) -> None:
        """Applies parameter updates directly to flat_params.

        Args:
            updates: The computed parameter updates.
        """
        self.flat_params.add_(-self.learning_rate * updates)

    @torch.no_grad()
    def _compute_jacobian(
        self,
        inputs: Any,
        targets: Any,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Any]:
        """Computes the Jacobian of the residuals with respect to model parameters.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, target_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple: A tuple containing:
                - jacobian: The Jacobian matrix of shape `(num_residuals, num_params)`.
                - residuals: Residual vector of shape `(num_residuals, 1)`.
                - outputs: Model outputs of shape `(batch_size, target_dim, ...)`.
        """
        buffers = dict(self._model.named_buffers())

        def compute_residuals(flat_params: Tensor, input: Any, target: Any) -> Tensor:
            if param_indices is not None:
                full_params = self.flat_params.clone()
                full_params[param_indices] = flat_params
            else:
                full_params = flat_params

            # Split flat_params into tensors matching the shapes of the model parameters
            param_list = torch.split(full_params, self._splits)

            # Map the split tensors back to their names
            params = {
                name: tensor.view_as(param)
                for (name, param), tensor in zip(self._params.items(), param_list)
            }

            params_and_buffers = {**params, **buffers}
            outputs = functional_call(self._model, params_and_buffers, input)
            return self.loss_fn.residuals(outputs, target)

        # Compute outputs and residuals for the full batch
        outputs = self.forward(inputs)
        residuals = self.loss_fn.residuals(outputs, targets)

        # Adjust flat_params to focus on the selected subset, if provided
        flat_params = (
            self.flat_params
            if param_indices is None
            else self.flat_params[param_indices]
        )

        jacobians: Tensor
        if self.use_vmap:
            # Compute per-sample Jacobian with vmap and jacrev
            jacobian_func = jacrev(compute_residuals)
            jacobians = vmap(
                jacobian_func, in_dims=(None, 0, 0), randomness='different'
            )(
                flat_params,
                tree_unsqueeze(inputs, dim=1),
                tree_unsqueeze(targets, dim=1),
            )
            jacobians = jacobians.squeeze(1)
        else:
            # Compute per-batch Jacobian with jacrev
            jacobian_func = jacrev(lambda p: compute_residuals(p, inputs, targets))
            jacobians = jacobian_func(flat_params)  # type: ignore

        # Flatten batches and outputs into a matrix to solve least-squares problems.
        residuals = residuals.view(-1, 1)
        jacobians = jacobians.view(-1, flat_params.numel())

        return jacobians, residuals, outputs

    @torch.no_grad()
    def _sliced_gauss_newton_overdetermined(
        self,
        inputs: Any,
        targets: Any,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Any]:
        """Gauss-Newton approximation for overdetermined systems using slicing.

        This method handles large overdetermined systems by dividing the input into
        smaller slices. For each slice, the Jacobian matrix is computed and used to
        incrementally build the full Gauss-Newton Hessian approximation and the
        right-hand side (RHS) vector.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, output_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple:
                - JJ: `(num_parameters, num_parameters) = J' * J`
                - rhs: `(num_parameters, 1) = J' * residuals`.
                - outputs: `(batch_size, output_dim, ...)`
        """
        assert self.max_batch_size is not None
        assert self._batch_size is not None

        batch_size = self._batch_size
        num_params = (
            param_indices.numel() if param_indices is not None else self._num_params
        )

        # Use one tensor from the inputs to obtain device and dtype.
        first_tensor = tree_first_tensor(inputs)
        device = first_tensor.device
        dtype = first_tensor.dtype

        JJ = torch.zeros((num_params, num_params), dtype=dtype, device=device)
        rhs = torch.zeros((num_params, 1), dtype=dtype, device=device)

        outputs_slices: list[Any] = []

        for start in range(0, batch_size, self.max_batch_size):
            end = min(start + self.max_batch_size, batch_size)
            inputs_slice = tree_slice(inputs, start, end)
            targets_slice = tree_slice(targets, start, end)

            J_slice, residuals_slice, outputs_slice = self._compute_jacobian(
                inputs_slice, targets_slice, param_indices
            )
            outputs_slices.append(outputs_slice)

            JJ += J_slice.t().matmul(J_slice)
            rhs += J_slice.t().matmul(residuals_slice)

        outputs = tree_cat(outputs_slices, dim=0)
        return JJ, rhs, outputs

    @torch.no_grad()
    def _sliced_gauss_newton_underdetermined(
        self,
        inputs: Any,
        targets: Any,
        param_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        """Gauss-Newton approximation for underdetermined systems using slicing.

        This method handles large underdetermined systems by dividing the input into
        smaller slices. For each slice, it computes the local Jacobian and residuals,
        concatenating them into a full J and residuals.

        Args:
            inputs: Input tensor `(batch_size, input_dim, ...)`.
            targets: Target tensor `(batch_size, output_dim, ...)`.
            param_indices: Indices of selected parameters if using a subset, else None.

        Returns:
            tuple:
                - J: Full Jacobian `(num_residuals, num_parameters)`
                - JJ: `(num_residuals, num_residuals) = J * J'`
                - rhs: `(num_residuals, 1) = residuals`
                - outputs: `(batch_size, output_dim, ...)`
        """
        assert self.max_batch_size is not None
        assert self._batch_size is not None

        batch_size = self._batch_size

        J_slices: list[Tensor] = []
        residuals_slices: list[Tensor] = []
        outputs_slices: list[Any] = []

        for start in range(0, batch_size, self.max_batch_size):
            end = min(start + self.max_batch_size, batch_size)
            inputs_slice = tree_slice(inputs, start, end)
            targets_slice = tree_slice(targets, start, end)

            J_slice, residuals_slice, outputs_slice = self._compute_jacobian(
                inputs_slice, targets_slice, param_indices
            )

            J_slices.append(J_slice)
            residuals_slices.append(residuals_slice)
            outputs_slices.append(outputs_slice)

        # Concatenate all slices to form the full J, residuals, and outputs
        J = torch.cat(J_slices, dim=0)
        residuals = torch.cat(residuals_slices, dim=0)
        outputs = tree_cat(outputs_slices, dim=0)

        # Compute JJ and rhs as in the non-sliced scenario for underdetermined case
        JJ = J @ J.t()  # JJ = J * J'
        rhs = residuals  # rhs = residuals
        return J, JJ, rhs, outputs

    @torch.no_grad()
    def training_step(
        self,
        inputs: Any,
        targets: Any,
    ) -> tuple[Any, Tensor, bool, dict[str, Any]]:
        """Performs a single training step.

        Args:
            inputs: Input tensor of shape `(batch_size, input_dim, ...)`.
            targets: Target tensor of shape `(batch_size, target_dim, ...)`.

        Returns:
            tuple: A tuple containing:
                - outputs: Model outputs for the given inputs.
                - loss: The computed loss value.
                - stop_training: Whether training should stop.
                - logs: Additional metadata (e.g., damping factor, attempts).
        """
        if self._batch_size is None:
            # Initialize during the first train step
            outputs = self._model(inputs)
            residuals = self.loss_fn.residuals(outputs, targets)
            self._batch_size = residuals.shape[0]
            self._num_residuals = residuals.numel()

        assert self._batch_size
        assert self._num_residuals

        batch_size = self._batch_size
        num_residuals = self._num_residuals
        num_params = self._num_params

        param_indices = None
        if self.param_selection_strategy is not None:
            param_indices = self.param_selection_strategy.select_parameters()
            num_params = param_indices.numel()

        overdetermined = num_residuals >= num_params

        if self.max_batch_size is not None and self.max_batch_size < batch_size:
            # reduced memory sliced computation
            if overdetermined:
                JJ, rhs, outputs = self._sliced_gauss_newton_overdetermined(
                    inputs, targets, param_indices
                )
                J = None
            else:
                J, JJ, rhs, outputs = self._sliced_gauss_newton_underdetermined(
                    inputs, targets, param_indices
                )
        else:
            J, residuals, outputs = self._compute_jacobian(
                inputs, targets, param_indices
            )
            if overdetermined:
                # overdetermined
                JJ = J.t() @ J  # JJ = J' * J
                rhs = J.t() @ residuals  # rhs = J' * residuals
            else:
                # underdetermined
                JJ = J @ J.t()  # JJ = J * J'
                rhs = residuals  # rhs = residuals

        # Normalize for numerical stability
        normalization_factor = 1.0 / batch_size
        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value
        loss = self.loss_fn(outputs, targets)

        stop_training = False
        attempt = 0
        self.damping_strategy.initialize_step(loss)

        while True:  # Infinite loop, break conditions inside
            params_updated = False

            # Try to update the parameters
            try:
                # Apply damping to the Gauss-Newton Hessian approximation
                JJ_damped = self.damping_strategy.apply(JJ)

                # Compute the updates:
                # - Overdetermined: updates = (J' * J + damping)^-1 * J'*residuals
                # - Underdetermined: updates = J' * (J * J' + damping)^-1 * residuals
                updates = self._solve(JJ_damped, rhs)

                if not overdetermined:
                    assert J is not None
                    updates = J.t().matmul(updates)

                updates = updates.view(-1)

                if param_indices is not None:
                    full_updates = torch.zeros(
                        self._num_params,
                        device=updates.device,
                        dtype=updates.dtype,
                    )
                    full_updates[param_indices] = updates
                    updates = full_updates

                # Check if updates are finite
                if torch.all(torch.isfinite(updates)):
                    params_updated = True
                    self._apply_updates(updates)

            except Exception as e:
                logger.warning(f'An exception occurred: {e}')

            if attempt < self.attempts_per_step:
                attempt += 1

                if params_updated:
                    # Compute the new loss value
                    new_outputs = self.forward(inputs)
                    new_loss = self.loss_fn(new_outputs, targets)

                    if new_loss < loss:
                        # Accept the new model parameters and backup them
                        loss = new_loss
                        self.damping_strategy.on_successful_update(loss)
                        self.backup_parameters()
                        break

                    # Restore the old parameters and try a new damping factor
                    self.restore_parameters()

                # Adjust the damping factor for the next attempt
                self.damping_strategy.on_unsuccessful_update(loss)

                # Check if should stop attempts and just take the update
                stop_attempts = self.damping_strategy.stop_attempts(loss)

                # Check if training should stop
                stop_training = self.damping_strategy.stop_training(loss)
                if stop_training or stop_attempts:
                    break
            else:
                break

        logs = {
            'damping': self.damping_strategy.get_current_damping(),
            'attempts': attempt,
        }
        # print('lm-train')

        return outputs, loss, stop_training, logs

    @property
    def model(self) -> torch.nn.Module:
        """The model being trained.

        Returns:
            torch.nn.Module: The neural network model used for training.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """The device on which the model's parameters are stored.

        This determines whether the model is on a CPU, GPU, or another device.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self._model.parameters()).device


class OptimizerModule(TrainingModule):
    """Train step for standard optimizers (e.g., Adam, SGD).

    This module provides a simple training loop for models using standard
    first-order optimization methods like SGD or Adam. It wraps the model,
    optimizer, and loss function, and provides a `training_step` method
    to perform parameter updates.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ) -> None:
        """Initializes the OptimizerModule.

        Args:
            model: The neural network model to be trained.
            optimizer: The optimizer used for training (e.g., SGD, Adam).
            loss_fn: The loss function used to compute the training objective.
        """
        self._model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def training_step(
        self,
        inputs: Any,
        targets: Any,
    ) -> tuple[Any, Tensor, bool, dict[str, Any]]:
        """Performs a training step using a standard optimizer.

        This method computes the loss for the given inputs and targets, performs
        backpropagation, and updates the model parameters using the optimizer.

        Args:
            inputs: Input tensor for the model, with shape depending on the task.
            targets: Target tensor, with shape depending on the task.

        Returns:
            tuple:
                - outputs: The model's predictions for the given inputs.
                - loss: The computed loss value.
                - stop_training: Always False, as it does not handle early stopping.
                - logs: An empty dictionary, as it does not provide additional logging.
        """
        # Forward pass
        outputs = self._model(inputs)
        loss: Tensor = self.loss_fn(outputs, targets)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return outputs, loss, False, {}

    @property
    def model(self) -> torch.nn.Module:
        """The model being trained.

        Returns:
            torch.nn.Module: The neural network model used for training.
        """
        return self._model

    @property
    def device(self) -> torch.device:
        """The device on which the model's parameters are stored.

        This determines whether the model is on a CPU, GPU, or another device.

        Returns:
            torch.device: The device of the model's parameters.
        """
        return next(self._model.parameters()).device


def tree_unsqueeze(tree: Any, dim: int = 0) -> Any:
    """Recursively unsqueeze every tensor in a pytree along the given dimension.

    Args:
        tree: The pytree containing tensors and non-tensor leaves.
        dim: The dimension along which to unsqueeze each tensor.

    Returns:
        A new pytree with every tensor unsqueezed along the specified dimension.
    """
    return tree_map(lambda x: x.unsqueeze(dim) if isinstance(x, Tensor) else x, tree)


def tree_to_device(tree: Any, device: torch.device | str) -> Any:
    """Recursively move all tensor leaves in a pytree to the specified device.

    Args:
        tree: The pytree containing tensors and non-tensor leaves.
        device: The target device (e.g. CPU or GPU) to move the tensors to.

    Returns:
        A new pytree with every tensor moved to the specified device.
    """
    return tree_map(lambda x: x.to(device) if isinstance(x, Tensor) else x, tree)


def tree_first_tensor(tree: Any) -> Tensor:
    """Return the first tensor found in the pytree.

    Args:
        tree: The pytree to search for a tensor.

    Returns:
        The first tensor encountered in the pytree.

    Raises:
        ValueError: If no tensor is found in the pytree.
    """
    flat_leaves, _ = tree_flatten(tree)
    first_tensor = next((x for x in flat_leaves if isinstance(x, Tensor)), None)
    if first_tensor is None:
        raise ValueError('No tensor found in the given pytree.')
    return first_tensor


def tree_cat(trees: list[Any], dim: int = 0) -> Any:
    """Concatenate a list of pytrees along the given dimension.

    Tensor leaves are concatenated using torch.cat; for non-tensor leaves only the
    first one is taken.

    Args:
        trees: A list of pytrees to concatenate.
        dim: The dimension along which to concatenate the tensor leaves.

    Returns:
        A new pytree with tensor leaves concatenated along the specified dimension.
    """
    # Flatten the first tree to get its structure.
    _, tree_def = tree_flatten(trees[0])
    # Flatten each tree to get the leaves, all trees must share the same structure.
    all_leaves = [tree_flatten(tree)[0] for tree in trees]
    cat_leaves = []
    for group in zip(*all_leaves):
        if isinstance(group[0], Tensor):
            cat_leaves.append(torch.cat(group, dim=dim))
        else:
            cat_leaves.append(group[0])
    return tree_unflatten(cat_leaves, tree_def)


def tree_slice(tree: Any, start: int, end: int) -> Any:
    """Slice every indexable leaf in the pytree from start to end.

    For tensors and sequence types (e.g., lists, tuples), standard slicing is applied.

    Args:
        tree: The pytree whose indexable leaves will be sliced.
        start: The start index of the slice.
        end: The end index of the slice.

    Returns:
        A new pytree with each indexable leaf sliced from start to end.
    """
    return tree_map(
        lambda x: (
            x[start:end] if hasattr(x, '__getitem__') and not isinstance(x, str) else x
        ),
        tree,
    )


def tree_indices(tree: Any, indices: list[int]) -> Any:
    """Select elements from every indexable leaf in the pytree using the provided list
    of indices.

    For any indexable object, this function first attempts to index directly with the
    list of indices. If that fails, it falls back to iterating over the indices
    and reconstructing the object.

    Args:
        tree: The pytree whose indexable leaves will be indexed.
        indices: A list of indices to select from each indexable leaf.

    Returns:
        A new pytree with each indexable leaf selected by the given indices.
    """
    return tree_map(
        lambda x: (
            x[indices]
            if hasattr(x, '__getitem__') and not isinstance(x, str)
            else type(x)(x[i] for i in indices)
        ),
        tree,
    )

class CustomLightningModule(pl.LightningModule):
    """PyTorch Lightning Module with support for custom TrainingModule and torchmetrics.

    This module integrates a `TrainingModule` for custom training logic and allows
    logging metrics using `torchmetrics`. It disables PyTorch Lightning's automatic
    optimization, enabling full control over the training loop.
    """

    def __init__(
        self,
        training_module: TrainingModule,
        metrics: dict[str, Metric] | None = None,
    ) -> None:
        """Initializes the CustomLightningModule.

        Args:
            training_module: An instance of `TrainingModule` encapsulating custom
                training logic.
            metrics: A dictionary of `torchmetrics.Metric` objects for evaluation,
                with metric names as keys. Defaults to an empty dictionary.
        """
        super().__init__()
        self.training_module = training_module
        self.metrics = metrics or {}
        self.automatic_optimization = False  # Disable automatic optimization

    def on_fit_start(self) -> None:
        """Moves metrics to the device where the model resides."""
        device = self.device  # Get the device of the model
        for metric in self.metrics.values():
            metric.to(device)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Performs a single training step.

        Args:
            batch: A tuple `(inputs, targets)` containing the input and target tensors.
            batch_idx: The index of the batch (required by PyTorch Lightning).

        Returns:
            The computed loss for the current batch.
        """
        inputs, targets = batch

        # Perform training step using the custom TrainingModule
        outputs, loss, stop_training, logs = self.training_module.training_step(
            inputs, targets
        )

        # # Convert logs into tensors for compatibility
        # logs = {
        #     key: (
        #         value
        #         if isinstance(value, Tensor)
        #         else torch.tensor(value, dtype=torch.float32)
        #     )
        #     for key, value in logs.items()
        # }

        # # Compute metrics if defined
        # metric_logs = {}
        # for name, metric in self.metrics.items():
        #     metric_logs[name] = metric(outputs, targets)

        # # Log metrics
        # self.log_dict(
        #     {name: value.item() for name, value in metric_logs.items()},
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=True,
        # )

        # # Log loss and additional logs
        # self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log_dict(logs, on_step=True, on_epoch=False, prog_bar=True)

        # Signal Lightning to stop training if necessary
        self.trainer.should_stop = stop_training

        return loss

    def configure_optimizers(self) -> list:
        """Prevents PyTorch Lightning from performing optimizer steps.

        Returns:
            An empty list as no optimizers are used in this module.
        """
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the model.

        Args:
            x: Input tensor to the model.

        Returns:
            The output of the model.
        """
        return self.training_module.model(x)


def fit(
    training_module: TrainingModule,
    dataloader: DataLoader,
    epochs: int,
    metrics: dict[str, Metric] | None = None,
    overwrite_progress_bar: bool = True,
    update_every_n_steps: int = 1,
) -> None:
    """Fit function with support for TrainingModule and torchmetrics.

    Trains the model for a specified number of epochs. It supports logging metrics using
    `torchmetrics` and provides detailed progress tracking using `tqdm`.

    Args:
        training_module: A `TrainingModule` encapsulating the training logic.
        dataloader: A PyTorch DataLoader.
        epochs: The number of epochs.
        metrics: Optional dict of torchmetrics.Metric objects.
        overwrite_progress_bar: If True, mimic a single-line progress bar similar
            to PyTorch Lightning (old bars overwritten).
        update_every_n_steps: Update the progress bar and displayed logs every n steps.
    """
    assert update_every_n_steps > 0
    device = training_module.device
    steps = len(dataloader)
    stop_training = False

    if metrics:
        metrics = {name: metric.to(device) for name, metric in metrics.items()}

    for epoch in range(epochs):
        if stop_training:
            break

        # Create a new progress bar for this epoch
        progress_bar = tqdm(
            total=steps,
            desc=f'Epoch {epoch + 1}/{epochs}',
            leave=not overwrite_progress_bar,  # Leave bar if overwrite is False
            dynamic_ncols=True,
        )
        total_loss = 0.0
        steps_since_update = 0

        for step, (inputs, targets) in enumerate(dataloader):
            # Ensure that inputs and targets are on the same device as the model
            inputs = tree_to_device(inputs, device)
            targets = tree_to_device(targets, device)

            # Perform a training step
            outputs, loss, stop_training, logs = training_module.training_step(
                inputs, targets
            )

            total_loss += loss.item()

            # Update metrics if provided
            if metrics:
                for name, metric in metrics.items():
                    metric(outputs, targets)

            # Format logs
            formatted_logs = {'loss': f'{loss:.4e}'}
            if metrics:
                for name, metric in metrics.items():
                    formatted_logs[name] = metric.compute().item()
            for key, value in logs.items():
                if isinstance(value, Tensor):
                    value = value.item()
                formatted_logs[key] = (
                    f'{value:.4e}' if isinstance(value, float) else str(value)
                )

            steps_since_update += 1
            if (
                steps_since_update == update_every_n_steps
                or step == steps - 1
                or stop_training
            ):
                # Update the progress bar and logs
                progress_bar.update(steps_since_update)
                progress_bar.set_postfix(formatted_logs)
                steps_since_update = 0

            if stop_training:
                # End early, ensure progress bar remains visible
                progress_bar.leave = True
                break

        # Reset metrics at the end of the epoch
        if metrics:
            for metric in metrics.values():
                metric.reset()

        # Epoch summary
        avg_loss = total_loss / steps
        if overwrite_progress_bar:
            progress_bar.set_postfix({'epoch_avg_loss': f'{avg_loss:.4e}'})
        else:
            progress_bar.write(
                f'Epoch {epoch + 1} complete. Average loss: {avg_loss:.4e}'
            )

        # Ensure the final progress bar is left visible
        if epoch == epochs - 1 or stop_training:
            progress_bar.leave = True

        progress_bar.close()

    # Final training summary
    if overwrite_progress_bar:
        print(f'Training complete. Final epoch average loss: {avg_loss:.4e}')


class FastDataLoader(DataLoader):
    """A lightweight and efficient data loader optimized for small datasets.

    This loader addresses the performance bottleneck caused by the overhead of
    `torch.utils.data.DataLoader` when dealing with small models or datasets that can
    fit entirely into RAM or GPU memory. The entire dataset is preloaded into memory and
    collated using a provided or default collate function.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        repeat: int = 1,
        shuffle: bool = False,
        device: torch.device | str = 'cpu',
        collate_fn: Callable[[list], Any] | None = None,
    ) -> None:
        """Initializes the FastDataLoader.

        Args:
            dataset: A PyTorch Dataset from which data will be extracted.
            batch_size: Number of samples per batch.
            repeat: Number of times to repeat the dataset.
            shuffle: If True, shuffle the data at the start of each repetition.
            device: The device on which to load the data.
            collate_fn: A function used to collate individual samples into a batch.
                        If None, the default_collate function is used.
        """
        super().__init__(dataset=dataset, batch_size=batch_size)
        self.repeat = repeat
        self.shuffle = shuffle

        # Build the sample list using indexing.
        self.examples = list(dataset)  # type: ignore
        self.num_examples = len(self.examples)

        # Use default_collate if no collate function is provided.
        if collate_fn is None:
            collate_fn = default_collate

        # Pre-collate the entire dataset and move it to the desired device.
        self.examples = collate_fn(self.examples)
        self.examples = tree_to_device(self.examples, device)

    def __iter__(self) -> Generator[Any, Any, None]:
        """Creates an iterator that yields batches of data.

        For each repetition, if shuffling is enabled, the dataset indices are shuffled
        before batching. Batches are then produced by selecting the appropriate indices
        from the pre-collated data.

        Yields:
            A batch of data from the pre-collated dataset.
        """
        assert self.batch_size
        for _ in range(self.repeat):
            indices = torch.arange(self.num_examples)
            if self.shuffle:
                indices = indices[torch.randperm(self.num_examples)]
            indices = indices.tolist()
            for i in range(0, self.num_examples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                yield tree_indices(self.examples, batch_indices)

    def __len__(self) -> int:
        """Returns the total number of batches across all repetitions.

        Returns:
            The number of batches in one epoch multiplied by the number of repetitions.
        """
        assert self.batch_size
        batches_per_epoch = (self.num_examples + self.batch_size - 1) // self.batch_size
        return self.repeat * batches_per_epoch
