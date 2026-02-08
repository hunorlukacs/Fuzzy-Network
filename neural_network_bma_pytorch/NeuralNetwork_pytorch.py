import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers=[]):
        """
        Initiates a Neural Network model.
        layers: (list) - Format: [[in_dim_1, out_dim_1, activation_1], [in_dim_2, out_dim_2, activation_2], ...]
        """
        super(NeuralNetwork, self).__init__()
        assert len(layers) > 0, (
            "[-] The length of the layers has to be greater than 0!\n"
            "[-] Format: [[in_dim_1, out_dim_1, activation_1], [in_dim_2, out_dim_2, activation_2], ...]"
        )

        modules = []
        for i, (in_dim, out_dim, activation) in enumerate(layers):
            modules.append(nn.Linear(in_dim, out_dim))
            # Only add activation if it is not 'linear' OR it is not the last layer
            if activation is not None and not (activation.lower() == 'linear' and i == len(layers)-1):
                modules.append(self._get_activation(activation))
        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def summary(self, input_size):
        """
        Prints a summary of the model like Keras.
        input_size: tuple -> shape of input (without batch dimension)
        """
        from torchsummary import summary
        summary(self, input_size)

    def get_trainable_params(self):
        """
        Returns flattened trainable parameters as numpy array.
        """
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params).cpu().numpy()

    def set_trainable_params(self, params):
        """
        Assign values from a 1D numpy array to the model’s parameters.
        """
        params = torch.tensor(params, dtype=torch.float32)
        start = 0
        for p in self.parameters():
            numel = p.numel()
            new_val = params[start:start+numel].view_as(p.data)
            p.data.copy_(new_val)
            start += numel

    def get_genes(self):
        """
        Returns the models' parameters as gene list (numpy array).
        """
        return self.get_trainable_params()

    def genes_len(self):
        return len(self.get_genes())

    def _get_activation(self, activation):
        """
        Map string to PyTorch activation function.
        """
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        elif activation == "linear" or activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
