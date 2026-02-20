import torch
import torch.nn as nn
import torch.nn.functional as F

class LipschitzLinear(nn.Module):
    """Linear layer with learnable Lipschitz constraint"""
    def __init__(self, in_features, out_features, init_ci='auto'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialize ci based on initial weight norm
        if init_ci == 'auto':
            with torch.no_grad():
                W = self.linear.weight.data
                absrowsum = torch.sum(torch.abs(W), dim=1)
                init_ci = absrowsum.max().item() * 2.0  # 2x initial norm
        
        # Learnable ci parameter (use raw value, apply softplus later)
        self.ci_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(init_ci)) - 1.0))
        
    def get_lipschitz_constant(self):
        """Get current Lipschitz constant (after softplus)"""
        return F.softplus(self.ci_raw)
    
    def forward(self, x):
        ci = F.softplus(self.ci_raw)
        
        # Normalize weights before forward pass
        W = self.linear.weight
        absrowsum = torch.sum(torch.abs(W), dim=1, keepdim=True)
        scale = torch.clamp(ci / (absrowsum + 1e-10), max=1.0)
        W_normalized = W * scale
        
        # Use normalized weights for forward pass
        return F.linear(x, W_normalized, self.linear.bias)


class LipschitzMLP(nn.Module):
    def __init__(self, layer_sizes, init_cis=None, lipschitz_lambda=0.001):
        """
        Parameters:
        -----------
        layer_sizes : list of int
            Network architecture, e.g., [20, 50, 30, 2]
        init_cis : None, 'auto', list, or float
            Initial Lipschitz constants per layer
            - None or 'auto': Initialize based on weight norms
            - float: Use same value for all layers
            - list: Per-layer values
        lipschitz_lambda : float
            Weight for Lipschitz regularization term
        """
        super().__init__()
        self.lipschitz_lambda = lipschitz_lambda
        
        # Handle init_cis
        if init_cis is None or init_cis == 'auto':
            init_cis = ['auto'] * (len(layer_sizes) - 1)
        elif isinstance(init_cis, (int, float)):
            init_cis = [init_cis] * (len(layer_sizes) - 1)
        
        # Build layers
        layers = []
        self.lipschitz_layers = []  # Keep track of Lipschitz layers
        
        for i in range(len(layer_sizes) - 1):
            lip_layer = LipschitzLinear(layer_sizes[i], layer_sizes[i+1], init_cis[i])
            layers.append(lip_layer)
            self.lipschitz_layers.append(lip_layer)
            
            if i < len(layer_sizes) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def lipschitz_regularization(self):
        """
        Compute Lipschitz regularization: λ * ∏ softplus(ci)
        This encourages the network to have a smaller Lipschitz bound
        """
        if self.lipschitz_lambda == 0:
            return 0.0
        
        lip_product = 1.0
        for layer in self.lipschitz_layers:
            lip_product = lip_product * layer.get_lipschitz_constant()
        
        return self.lipschitz_lambda * lip_product
    
    def get_lipschitz_constants(self):
        """Get current Lipschitz constant for each layer"""
        return [layer.get_lipschitz_constant().item() for layer in self.lipschitz_layers]
    
    def get_lipschitz_bound(self):
        """Get overall network Lipschitz bound (product of per-layer constants)"""
        bound = 1.0
        for ci in self.get_lipschitz_constants():
            bound *= ci
        return bound