import torch
import torch.nn as nn
import numpy as np
import copy
from fuzzy_network_pytorch.FuzzyLayer import FuzzyLayer
from fuzzy_network_pytorch.FuzzyLayer_Gauss import FuzzyLayer_Gauss
from datetime import datetime



class FuzzyNetwork(nn.Module):
    """
    A multi-layer fuzzy inference network built from stacked FuzzyLayer modules.
    Each layer transforms input fuzzy features through fuzzy rule-based inference.
    """

    def __init__(self,
                 f_layers=[],
                 ante_memb='trap',
                 cons_memb='trap',
                 device='cpu'):
        """
        Initialize a Fuzzy Network.

        Parameters
        ----------
        f_layers : list of lists
            Format: [[in_dim, out_dim, nr_rules], ...]
        ante_memb : "trap" or "gauss"
        cons_memb : "trap" or "gauss"
        """
        super(FuzzyNetwork, self).__init__()

        assert len(f_layers) > 0, (
            "[-] f_layers must contain at least one fuzzy layer definition!"
        )

        self.ante_memb = ante_memb
        self.cons_memb = cons_memb
        self.device = device

        self.f_layers = nn.ModuleList()
        for layer_cfg in f_layers:
            in_dim, out_dim, nr_rules = layer_cfg
            self.attach_layer(in_dim, out_dim, nr_rules, self.device)

    def attach_layer(self, in_dim, out_dim, nr_rules, device):
        """Attach a new fuzzy layer using the new FuzzyLayer."""
        new_layer = FuzzyLayer_Gauss(
            in_dim=in_dim,
            out_dim=out_dim,
            nr_rules=nr_rules,
            ante_type=self.ante_memb,
            cons_type=self.cons_memb,
            device=device,
        )
        # new_layer = FuzzyLayer(
        #     in_dim=in_dim,
        #     out_dim=out_dim,
        #     nr_rules=nr_rules,
        #     device=device,
        # )
        self.f_layers.append(new_layer)

    def forward(self, x):
        """Forward pass through the fuzzy network."""
        for layer in self.f_layers:
            x = layer(x)
        return x

    def summary(self, show_params=False):
        """Print a layer-by-layer summary of the network."""
        print("Fuzzy Network Summary:")
        print("======================")
        for i, layer in enumerate(self.f_layers):
            print(f"Layer {i}:")
            print(f"  Type: {type(layer).__name__}")
            print(f"  Input Dimension: {layer.in_dim}")
            print(f"  Output Dimension: {layer.out_dim}")
            print(f"  Number of Rules: {layer.nr_rules}")
            if show_params and hasattr(layer, "summary"):
                layer.summary(show_params=True)
        print("======================")

    def get_trainable_params(self):
        """Return all trainable parameters as a flattened NumPy array."""
        params = []
        for p in self.parameters():
            params.append(p.detach().cpu().view(-1))
        return torch.cat(params).numpy()

    def set_trainable_params(self, params):
        """Set model parameters from a flattened array."""
        start = 0
        params_tensor = torch.tensor(params, dtype=torch.float32)
        for p in self.parameters():
            length = p.numel()
            new_vals = params_tensor[start:start + length].view_as(p)
            p.data.copy_(new_vals)
            start += length

    def get_genes(self):
        """Return parameters reshaped into trapezoidal gene list."""
        flat_params = self.get_trainable_params()
        return flat_params.reshape((-1, 4))

    def genes_len(self):
        """Return number of trapezoidal genes."""
        return len(self.get_genes())

    def plot_network(self, obs=None, PADDING_RATE=0):
        """Plot fuzzy rules for each layer."""
        for idx, layer in enumerate(self.f_layers):
            print(f"Plotting Layer {idx + 1}/{len(self.f_layers)}")
            layer.plot_rules(obs=obs)

            if hasattr(layer, "inference"):
                obs = layer.inference(obs)
                print(f"Output from Layer {idx + 1}: {obs.detach().cpu().numpy()}")

    def save_model(self, filepath=None):
        """Save the model state dictionary with timestamp."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"fuzzy_network_{timestamp}.pt"
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, map_location=None):
        """Load model weights from file."""
        state_dict = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filepath}")
        return self
