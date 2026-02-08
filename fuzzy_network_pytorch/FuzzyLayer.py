import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os



# ===============================
#  Trapezoidal Membership Function (vectorized)
# ===============================
def trapmf(x, abcd, eps=1e-8):
    """
    Vectorized trapezoidal membership.
    x : tensor broadcastable with abcd[..., :1]
         Typical shapes:
           x: (batch, in_dim) or (1, batch, in_dim) or (n_rules, batch, in_dim)
           abcd: (..., 4) where last dim is [a,b,c,d]
    Returns membership values with x/abcd broadcasted; returned shape = broadcast(x, abcd_without_last_dim)
    """
    # ensure last dim of abcd is 4
    assert abcd.shape[-1] == 4, "abcd must have last dim = 4"

    a = abcd[..., 0:1]
    b = abcd[..., 1:2]
    c = abcd[..., 2:3]
    d = abcd[..., 3:4]

    # x and abcd should be float tensors
    x = x.to(dtype=torch.float32)
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    c = c.to(dtype=torch.float32)
    d = d.to(dtype=torch.float32)

    # Broadcasted piecewise calculation
    y = torch.zeros_like(x + a, dtype=torch.float32)  # use x+a to get expanded shape
    # rising edge: a < x < b
    mask_rise = (a < x) & (x < b)
    y = torch.where(mask_rise, (x - a) / (b - a + eps), y)
    # top: b <= x <= c
    mask_top = (b <= x) & (x <= c)
    y = torch.where(mask_top, torch.ones_like(y), y)
    # falling edge: c < x < d
    mask_fall = (c < x) & (x < d)
    y = torch.where(mask_fall, (d - x) / (d - c + eps), y)

    # anything else remains 0
    return y

# ===============================
#  Mamdani with separated Antes / Cons
# ===============================
def mamdaniInference(Antes: torch.Tensor, Cons: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized Mamdani inference.
    
    Antes: (n_rules, in_dim, 4)
    Cons:  (n_rules, out_dim, 4)
    inputs: (batch, in_dim)
    
    Returns: (batch, out_dim)
    """
    device = inputs.device
    Antes = Antes.to(device)
    Cons = Cons.to(device)

    n_rules, in_dim, _ = Antes.shape
    batch = inputs.shape[0]
    out_dim = Cons.shape[1]

    # --------------------------
    # Vectorized membership degree
    # --------------------------
    # inputs: (batch, in_dim) → (1, batch, in_dim, 1)
    x_exp = inputs.unsqueeze(0).unsqueeze(-1)  # (1, batch, in_dim, 1)
    abcd_exp = Antes.unsqueeze(1)              # (n_rules, 1, in_dim, 4)
    memb = trapmf(x_exp, abcd_exp).squeeze(-1) # (n_rules, batch, in_dim)
    w_mins = torch.min(memb, dim=-1).values    # (n_rules, batch)

    # --------------------------
    # Expand w_mins to match Cons: (batch, n_rules, out_dim, 1)
    # --------------------------
    w = w_mins.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (batch, n_rules, 1, 1)
    Cons_exp = Cons.unsqueeze(0).expand(batch, -1, -1, -1) # (batch, n_rules, out_dim, 4)
    
    A = Cons_exp[..., 0:1]  # (batch, n_rules, out_dim, 1)
    B = Cons_exp[..., 1:2]
    C = Cons_exp[..., 2:3]
    D = Cons_exp[..., 3:4]

    eps = torch.finfo(torch.float32).eps

    # --------------------------
    # Mamdani terms (fully vectorized)
    # --------------------------
    term1 = 3 * w * (D**2 - A**2) * (1 - w)
    term2 = 3 * (w**2) * (C*D - A*B)
    term3 = (w**3) * (C - D + A - B) * (C - D - A + B)

    num = term1 + term2 + term3
    den = 2*w*(D-A) + (w**2)*(C + A - D - B) + eps

    # --------------------------
    # Sum over rules (dim=1)
    # --------------------------
    num_sum = num.sum(dim=1).squeeze(-1)  # (batch, out_dim)
    den_sum = den.sum(dim=1).squeeze(-1)

    Ys = torch.where(den_sum != 0, num_sum / den_sum, torch.zeros_like(num_sum, device=device))
    Ys = Ys / 3.0

    return Ys  # (batch, out_dim)


# ===============================
#  Constraint: ensure increasing order
# ===============================
class BreakpointIncreasingOrderConstraint(nn.Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 2.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, w):
        w = torch.clamp(w, self.min_val, self.max_val)

        # reshape to (N, 4)
        w_flat = w.view(-1, 4)

        w_sorted, _ = torch.sort(w_flat, dim=-1)

        return w_sorted.view_as(w)



# ===============================
#  FuzzyLayer (PyTorch) - unchanged API but uses vectorized inference
# ===============================
class FuzzyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, nr_rules, device='cuda', constraint_min=-1, constraint_max=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nr_rules = nr_rules
        self.device = torch.device(device)
        self.constraint = BreakpointIncreasingOrderConstraint(min_val=constraint_min, max_val=constraint_max)

        # Parameter shapes:
        # Antes: (n_rules, in_dim, 4)
        # Cons : (n_rules, out_dim, 4)
        self.Antes = nn.Parameter(torch.empty((nr_rules, in_dim, 4), device=self.device).uniform_(constraint_min, constraint_max))
        self.Cons = nn.Parameter(torch.empty((nr_rules, out_dim, 4), device=self.device).uniform_(constraint_min, constraint_max))

        # apply constraint initially
        with torch.no_grad():
            self.Antes.copy_(self.constraint(self.Antes))
            # Cons is (n_rules, out_dim, 4) -> apply constraint per 4-block
            self.Cons.copy_(self.constraint(self.Cons.view(-1, 4)).view_as(self.Cons))

    def forward(self, x):
        """Forward pass through the fuzzy layer with constraint enforcement."""
        return self.inference(x)
    
    def inference(self, x):
        # ensure input is batch x in_dim
        if x.dim() == 1:
            x_in = x.unsqueeze(0)
        else:
            x_in = x
        x_in = x_in.to(self.device)

        Antes = self.Antes.to(self.device)
        Cons = self.Cons.to(self.device)

        # returns (batch, out_dim)
        return mamdaniInference(Antes, Cons, x_in)

    # def apply_constraints(self):
    #     # move constraints to GPU
    #     self.Antes.data = self.constraint(self.Antes.data)
    #     self.Cons.data = self.constraint(self.Cons.data)
    def apply_constraints(self):
        with torch.no_grad():
            self.Antes.copy_(self.constraint(self.Antes))
            self.Cons.copy_(self.constraint(self.Cons.view(-1, 4)).view_as(self.Cons))

    def get_rule(self, rule_nr, cons_nr=None):
        if rule_nr < 0 or rule_nr >= self.nr_rules:
            raise ValueError(f"Rule {rule_nr} out of bounds")
        ante = self.Antes[rule_nr].detach().cpu().numpy()
        if cons_nr is None:
            cons = self.Cons[rule_nr].detach().cpu().numpy()
        else:
            cons = self.Cons[rule_nr, cons_nr].detach().cpu().numpy()
        return np.concatenate((ante, [cons]), axis=0)

    def summary(self, show_params=False):
        print("FuzzyLayer Summary:")
        print(f"{'Input Dimension:':<20} {self.in_dim}")
        print(f"{'Output Dimension:':<20} {self.out_dim}")
        print(f"{'Number of Rules:':<20} {self.nr_rules}")
        print(f"{'Antes shape:':<20} {tuple(self.Antes.shape)}")
        print(f"{'Cons shape:':<20} {tuple(self.Cons.shape)}")
        if show_params:
            print("Antes:", self.Antes)
            print("Cons:", self.Cons)

    def plot_rules(self, obs=None, max_rules_to_plot=21):
        Antes = self.Antes.detach().cpu().numpy()
        Cons = self.Cons.detach().cpu().numpy()

        nr_rules, nr_antecedents, _ = Antes.shape
        nr_consequents = Cons.shape[1]
        boundaries = np.array([[-1, 2]] * (nr_antecedents + nr_consequents))

        # limit plotting to avoid huge figures
        rules_to_plot = min(nr_rules, max_rules_to_plot)
        fig, ax = plt.subplots(nrows=rules_to_plot, ncols=nr_antecedents + nr_consequents,
                               figsize=(5 * (nr_antecedents + nr_consequents), 3 * rules_to_plot))
        if rules_to_plot == 1:
            ax = np.expand_dims(ax, 0)

        for plot_rule_idx in range(rules_to_plot):
            for feat_idx in range(nr_antecedents):
                x = np.linspace(boundaries[feat_idx, 0], boundaries[feat_idx, 1], 500)
                y = trapmf(torch.tensor(x, dtype=torch.float32), torch.tensor(Antes[plot_rule_idx, feat_idx]))
                ax[plot_rule_idx, feat_idx].plot(x, y)
                ax[plot_rule_idx, feat_idx].set_title(f'Antecedent[{plot_rule_idx},{feat_idx}]')
            for feat_idx in range(nr_consequents):
                x = np.linspace(boundaries[-1, 0], boundaries[-1, 1], 500)
                y = trapmf(torch.tensor(x, dtype=torch.float32), torch.tensor(Cons[plot_rule_idx, feat_idx]))
                ax[plot_rule_idx, nr_antecedents + feat_idx].plot(x, y)
                ax[plot_rule_idx, nr_antecedents + feat_idx].set_title(f'Consequent[{plot_rule_idx},{feat_idx}]')

        plt.tight_layout()
        plt.show()
