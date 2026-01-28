import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal, _make_orthogonal

class Reservoir(nn.Module):
    def __init__(self, 
                 res_dim, 
                 input_dim = None,
                 weight = None,
                 bias = True,
                 bias_scale = 0.1,
                 spectral_radius = 1,
                 det_norm = None,
                 leak_rate = .4,
                 sparsity = 0,
                 powerlaw_alpha = 1.75,
                 seed = 42,
                 **activation_kwargs):  
        
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = seed
        self.res_dim = res_dim
        self.input_dim = input_dim if input_dim is not None else res_dim

        # instantiate random weight matrix: input_dim -> res_dim
        if weight is None:
            weight = self.random_powerlaw_matrix(self.input_dim,
                                                 out_dim=res_dim,
                                                 alpha = powerlaw_alpha,
                                                 normalize_radius_to = spectral_radius,
                                                 normalize_det_to = det_norm,
                                                 sparsity = sparsity)
            initial_bias = 2 * torch.rand(res_dim) - 1
            bias_sum = initial_bias.sum()
            initial_bias -= bias_sum / res_dim
            bias = initial_bias * bias_scale
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", torch.zeros(res_dim))

        self.register_buffer("weight", weight)

        #instantiate random adjacency matrix with normalized spectral radius
        adjacency = torch.randint(0, 2, size=(res_dim, res_dim), dtype=torch.float32)
        # Normalize adjacency matrix spectral radius to prevent divergence
        eigenvalues = torch.linalg.eigvalsh(adjacency)
        spectral_radius_adj = torch.max(torch.abs(eigenvalues))
        if spectral_radius_adj > 0:
            adjacency = adjacency / (spectral_radius_adj + 0.1)  # Scale to < 1 for stability
        self.register_buffer("adjacency", adjacency)
        
        #leak rate $\gamma
        self.leak_rate = leak_rate
        #activation function - default to absolute tanh for stability
        self.activation = lambda x: torch.abs(torch.tanh(x))
        #prev output of all nodes - will be initialized dynamically in forward
        self.state = None

    def forward(self, x, n_steps = 1):
        with torch.no_grad():
            # Ensure x is 2D: (batch_size, input_dim)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            batch_size = x.shape[0]
            
            # Initialize state if needed or reset to match batch size
            if self.state is None or self.state.shape[0] != batch_size:
                self.state = torch.zeros((batch_size, self.res_dim), dtype=x.dtype, device=x.device)
            
            for step in range(n_steps):
                # x @ self.weight: (batch_size, input_dim) @ (input_dim, res_dim) -> (batch_size, res_dim)
                # self.state @ self.adjacency: (batch_size, res_dim) @ (res_dim, res_dim) -> (batch_size, res_dim)
                y = x @ self.weight + self.state @ self.adjacency + self.bias
                y = self.leak_rate * self.state + (1 - self.leak_rate) * self.activation(y)
                self.state = y
        
        return y
    
    def process_input(self, input_data: torch.Tensor, n_steps: int = 1, **kwargs) -> torch.Tensor:
        '''
        Process input data through the reservoir.
        Standard interface for edge server integration.
        
        Args:
            input_data: Tensor of shape (batch_size, input_dim)
            n_steps: Number of reservoir iterations
            **kwargs: Additional arguments (unused, for interface compatibility)
        
        Returns:
            Reservoir output tensor of shape (batch_size, res_dim)
        '''
        return self.forward(input_data, n_steps=n_steps)

    def powerlaw_random(self, 
                        dim, 
                        alpha = 1.75, 
                        x_min: float = 1):
        '''
        Sample numbers from a powerlaw
        '''
        rands = torch.rand(dim)
        out = x_min * (1 - rands) ** (-1 / (alpha - 1))
        return out
    
    def random_powerlaw_matrix(self,
                               dim, 
                               out_dim = None, 
                               alpha = 1.75,
                               x_min = 0.1,
                               normalize_det_to = None, 
                               normalize_radius_to = 1,
                               sparsity = 0.0):
        '''
        Generates random weights of the reservoir in a powerlaw matrix    
        '''
        if out_dim is None:
            out_dim = dim

        diagonal = self.powerlaw_random(out_dim, alpha = alpha, x_min = x_min)

        if normalize_det_to is not None:
            log_det = torch.log(diagonal).sum()
            det_n = torch.exp(log_det / out_dim)
            diagonal *= normalize_det_to / det_n
        elif normalize_radius_to is not None:
            radius = torch.max(diagonal)
            diagonal *= normalize_radius_to / radius

        # make some entries negative
        negative_mask = torch.rand(out_dim) > 0.5
        diagonal[negative_mask] = -diagonal[negative_mask]
        matrix = torch.diag(diagonal)  # (out_dim, out_dim)
        sparsity = min(1, sparsity)
        # Create random matrix (out_dim, dim) and make it orthogonal
        rot = _make_orthogonal(torch.randn(out_dim, dim) * (torch.rand(out_dim, dim) > sparsity))  # (out_dim, dim)
        # Apply rotation: rot @ matrix @ rot.T would give (out_dim, out_dim)
        # We want (dim, out_dim), so compute: rot.T @ matrix @ rot = (dim, out_dim) @ (out_dim, out_dim) @ (out_dim, dim) = (dim, dim)
        # Instead: return rot.T which is (dim, out_dim), but scaled/projected through matrix
        # Simple approach: just return rot.T scaled by diagonal
        return rot.T * torch.sqrt(diagonal.abs()).unsqueeze(0)


class Readout(nn.Module):
    def __init__(self,
                 res_dim: int,
                 io_dim: int,
                 lr: float = 0.001):
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.readout = nn.Linear(res_dim, io_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.readout.parameters(), lr)
    
    def forward(self, x):
        '''
        Forward pass through readout layer
        x: tensor of shape (batch_size, res_dim)
        returns: tensor of shape (batch_size, io_dim)
        '''
        return self.readout(x)

