import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal, _make_orthogonal

class Reservoir(nn.Module):
    def __init__(self, res_dim,
                #  input_dim = None, not used in this implementation
                #  ouput_dim = None,
                 weight = None,
                 bias = True,
                 bias_scale = 0.1,
                 spectral_radius = 1,
                 det_norm = None,
                 leak_rate = .2,
                 sparsity = 0,
                 powerlaw_alpha = 1.75,
                 seed = 42,
                 **activation_kwargs):  
        
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = seed 

        # instantiate random weight matrix
        if weight is None:
            weight = random_powerlaw_matrix(res_dim,
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

        #instantiate random adjacency matrix
        self.adjacency = np.random.randint(2,size=(res_dim,res_dim))
        
        #leak rate $\gamma
        self.leak_rate = leak_rate
        #activation function plus kwargs for activation function
        self.activation = activation(**activation_kwargs)
        #prev output of all nodes
        self.state = np.zeroes((res_dim,res_dim), dtype=np.float64)

    def forward(self, x, n_steps = 1):
        with torch.no_grad():
            for _ in range(n_steps):
                y = x @ self.weight + self.state @ self.adjacency + self.bias
                y = (1-self.leak_rate)* self.state \
                    + self.leak_rate*self.activation(y)
                self.state = y
        return y
    
    @staticmethod
    def powerlaw_random(dim, alpha = 1.75, 
                        x_min = 1):
        '''
        Sample numbers from a powerlaw
        '''
        rands = torch.rand(dim)
        out = x_min * (1 - rands) ** (-1 / (alpha - 1))
        return out
    
    @staticmethod
    def random_powerlaw_matrix(dim, 
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

        diagonal = Reservoir.powerlaw_random(out_dim, alpha = alpha, x_min = x_min)

        if normalize_det_to is not None:
            log_det = torch.log(diagonal).sum()
            det_n = torch.exp(log_det / dim)
            diagonal *= normalize_det_to / det_n
        elif normalize_radius_to is not None:
            radius = torch.max(diagonal)
            diagonal *= normalize_radius_to / radius

        # make some entries negative
        negative_mask = torch.rand(out_dim) > 0.5
        diagonal[negative_mask] = -diagonal[negative_mask]
        matrix = torch.diag(diagonal)
        sparsity = min(1, sparsity)
        rot = _make_orthogonal(torch.randn(out_dim, dim) * (torch.rand(out_dim, dim) > sparsity))
        return rot.T @ matrix @ rot

    

class Readout(nn.Module):
    def __init__(self,
                 res_dim,
                 io_dim):
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.readout = nn.linear(res_dim,io_dim).to(self.device)
