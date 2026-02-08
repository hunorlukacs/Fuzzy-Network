import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
#  Trapezoidal Membership Function
# ==========================================================
def trapmf(x, abcd, eps=1e-8):
    assert abcd.shape[-1] == 4
    
    # Sort parameters to ensure a <= b <= c <= d
    abcd_sorted = torch.sort(abcd, dim=-1).values 
    
    a = abcd_sorted[..., 0:1]
    b = abcd_sorted[..., 1:2]
    c = abcd_sorted[..., 2:3]
    d = abcd_sorted[..., 3:4]

    x = x.to(torch.float32)
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    c = c.to(torch.float32)
    d = d.to(torch.float32)

    y = torch.zeros_like(x + a, dtype=torch.float32)

    # 1. Rising slope
    mask_rise = (a < x) & (x < b)
    y = torch.where(mask_rise, (x - a) / (b - a + eps), y)

    # 2. Top plateau
    mask_top = (b <= x) & (x <= c)
    y = torch.where(mask_top, torch.ones_like(y), y)

    # 3. Falling slope
    mask_fall = (c < x) & (x < d)
    y = torch.where(mask_fall, (d - x) / (d - c + eps), y)

    return y


# ==========================================================
#  Gaussian Membership Function
# ==========================================================
def gaussmf(x, params, eps=1e-8):
    assert params.shape[-1] == 2
    mean = params[..., 0:1]
    sigma = params[..., 1:2]

    x = x.to(torch.float32)
    mean = mean.to(torch.float32)
    sigma = sigma.to(torch.float32)
    
    sigma = torch.abs(sigma)
    sigma = torch.clamp(sigma, min=eps)

    return torch.exp(-0.5 * ((x - mean) / sigma)**2)

# ==========================================================
#  Mamdani Inference
# ==========================================================
def mamdani_inference(antes, cons, inputs,
                      ante_type="trap",
                      cons_type="trap"):
    device = inputs.device
    antes = antes.to(device)
    cons = cons.to(device)

    nr_rules, in_dim, _ = antes.shape
    batch = inputs.shape[0]
    out_dim = cons.shape[1]

    # 1) Antecedent Membership
    x_exp = inputs.unsqueeze(0).unsqueeze(-1)

    if ante_type == "trap":
        memb = trapmf(x_exp, antes.unsqueeze(1)).squeeze(-1)
    elif ante_type == "gauss":
        memb = gaussmf(x_exp, antes.unsqueeze(1)).squeeze(-1)
    else:
        raise ValueError("ante_type must be 'trap' or 'gauss'")

    # Rule firing strength (w_mins) via Minimum operator
    w_mins = torch.min(memb, dim=-1).values 

    # 2) Expand for consequents
    w = w_mins.permute(1, 0).unsqueeze(-1).unsqueeze(-1) 
    cons_exp = cons.unsqueeze(0).expand(batch, -1, -1, -1)

    eps = torch.finfo(torch.float32).eps

    # Case 1: Trapezoidal consequents
    if cons_type == "trap":
        # Sort consequent parameters
        cons_sorted, _ = torch.sort(cons_exp, dim=-1)

        A = cons_sorted[..., 0:1]
        B = cons_sorted[..., 1:2]
        C = cons_sorted[..., 2:3]
        D = cons_sorted[..., 3:4]
        
        # Defuzzification (Centroid)
        term1 = 3 * w * (D**2 - A**2) * (1 - w)
        term2 = 3 * (w**2) * (C * D - A * B)
        term3 = (w**3) * (C - D + A - B) * (C - D - A + B)

        num = term1 + term2 + term3
        den = 2*w*(D-A) + (w**2)*(C + A - D - B) + eps

        num = num.sum(dim=1).squeeze(-1)
        den = den.sum(dim=1).squeeze(-1)

        return (num / den) / 3.0

    # Case 2: Gaussian consequents
    if cons_type == "gauss":
        means = cons_exp[..., 0] 
        w_squeezed = w.squeeze(-1)
        return (w_squeezed * means).sum(dim=1) / (w_squeezed.sum(dim=1) + eps)

    else:
        raise ValueError("cons_type must be 'trap' or 'gauss'")

# ==========================================================
# Fuzzy Layer Class
# ==========================================================
class FuzzyLayer_Gauss(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 nr_rules,
                 ante_type="gauss",
                 cons_type="trap",
                 device='cuda:0' if torch.cuda.is_available() else 'cpu'):

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nr_rules = nr_rules
        self.ante_type = ante_type
        self.cons_type = cons_type
        self.device = device
        print(self.device)
        ante_p = 4 if ante_type == "trap" else 2
        cons_p = 4 if cons_type == "trap" else 2

        self.Antes = nn.Parameter(
            torch.empty((nr_rules, in_dim, ante_p), device=self.device).uniform_(-1, 1)
        )
        self.Cons = nn.Parameter(
            torch.empty((nr_rules, out_dim, cons_p), device=self.device).uniform_(-1, 1)
        )

    def forward(self, x):
        x_in = x if x.dim() == 2 else x.unsqueeze(0)
        return mamdani_inference(self.Antes, self.Cons, x_in,
                                 ante_type=self.ante_type,
                                 cons_type=self.cons_type)

    def summary(self):
        print("=== Fuzzy Layer Summary ===")
        print("Input dim:", self.in_dim)
        print("Output dim:", self.out_dim)
        print("Rules:", self.nr_rules)
        print("Antecedent MF:", self.ante_type)
        print("Consequent MF:", self.cons_type)
        print("Antes shape:", tuple(self.Antes.shape))
        print("Cons shape:", tuple(self.Cons.shape))

    def plot_rules(self, obs=None, max_rules_to_plot=21):
        """
        Plot fuzzy rules for each layer, including antecedents and consequents.
        """
        Antes = self.Antes.detach().cpu()
        Cons = self.Cons.detach().cpu()

        nr_rules, n_ante, p = Antes.shape
        n_cons = Cons.shape[1]
        
        # ---- OBSERVATIONS ----
        if obs is not None:
            if not torch.is_tensor(obs):
                obs = torch.tensor(obs, dtype=torch.float32)
            
            if obs.shape[-1] != n_ante:
                print(f"Warning: Observation has {obs.shape[-1]} features, but layer expects {n_ante}.")

            obs_min = obs.min(dim=0).values - 1.0
            obs_max = obs.max(dim=0).values + 1.0
            colors = plt.cm.viridis(np.linspace(0, 1, len(obs)))
        else:
            colors = None

        # ---- PLOT GRID ----
        plot_rules = min(nr_rules, max_rules_to_plot)
        fig, ax = plt.subplots(plot_rules, n_ante + n_cons,
                            figsize=(4*(n_ante+n_cons), 3*plot_rules))

        if plot_rules == 1: ax = np.expand_dims(ax, 0)
        if (n_ante + n_cons) == 1: ax = np.expand_dims(ax, 1)
        if ax.ndim == 1: ax = np.expand_dims(ax, 0)

        def plot_mf(ax_obj, x, y, title):
            ax_obj.plot(x.numpy(), y.numpy())
            ax_obj.set_ylim([-0.05, 1.05])
            ax_obj.set_title(title)

        # ---- PLOT RULES ----
        for r in range(plot_rules):
            # ANTECEDENTS
            for i in range(n_ante):
                low, high = -2, 2 
                x = torch.linspace(low, high, 1000)
                params = Antes[r,i]

                if p == 2: # Gaussian
                    mean, sigma = params[0].item(), abs(params[1].item())
                    sigma = max(sigma, 1e-4)
                    y = torch.exp(-0.5 * ((x - mean)/sigma)**2)
                else: # Trap
                    y = trapmf(x, params)

                plot_mf(ax[r,i], x, y, f"Rule {r} - Antecedent {i}")

                if obs is not None:
                    for obs_idx, o in enumerate(obs):
                        x_obs = o[i].item()
                        if p == 2:
                             mean_t, sigma_t = torch.tensor(mean), torch.tensor(sigma)
                             mu = torch.exp(-0.5 * ((torch.tensor(x_obs) - mean_t) / sigma_t) ** 2).item()
                        else:
                             mu = trapmf(torch.tensor([x_obs]), params)[0].item()

                        ax[r,i].vlines(x_obs, 0, mu, color=colors[obs_idx], linestyles='dashed')
                        ax[r,i].plot(x_obs, 0, 'o', color=colors[obs_idx])
                        ax[r,i].hlines(mu, x_obs, high, color=colors[obs_idx], linestyles='dashed')

            # CONSEQUENTS
            for j in range(n_cons):
                low, high = -10, 10 
                x = torch.linspace(low, high, 1000)
                params = Cons[r,j]

                if params.numel() == 2: # Gaussian
                    mean, sigma = params[0].item(), abs(params[1].item())
                    sigma = max(sigma, 1e-4)
                    y = torch.exp(-0.5 * ((x - mean)/sigma)**2)
                else: # Trap
                    y = trapmf(x, params)

                plot_mf(ax[r, n_ante+j], x, y, f"Rule {r} - Consequent {j}")

                if obs is not None:
                    for obs_idx, o in enumerate(obs):
                        mu_list = []
                        for i in range(n_ante):
                            x_obs = o[i].item()
                            if p == 2:
                                mean_i = Antes[r,i][0].item()
                                sigma_i = max(abs(Antes[r,i][1].item()), 1e-4)
                                mu_i = np.exp(-0.5 * ((x_obs - mean_i)/sigma_i)**2)
                            else:
                                mu_i = trapmf(torch.tensor([x_obs]), Antes[r,i])[0].item()
                            mu_list.append(mu_i)

                        w = min(mu_list) 
                        if w < 1e-8: continue 

                        if params.numel() == 2: # Gaussian
                             mean_c = params[0].item()
                             sigma_c = max(abs(params[1].item()), 1e-4)
                             if w >= 1.0: delta = 0
                             else: delta = sigma_c * np.sqrt(-2.0 * np.log(w))
                             left, right = mean_c - delta, mean_c + delta
                        else: # Trap
                             a,b,c,d = torch.sort(params).values.tolist()
                             left = w*(b - a) + a
                             right = d - w*(d - c)

                        ax[r, n_ante+j].hlines(w, left, right, color=colors[obs_idx], linestyles='dashed')

        plt.tight_layout()
        plt.show()