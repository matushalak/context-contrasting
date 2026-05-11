import torch
import torch.nn as nn
from typing import Literal
from context_contrasting.mini_network.utils import EMA
from context_contrasting.utils import nonnegative, nonpositive, ThresholdReLU

class TinyCClayer(nn.Module):
    def __init__(self,
                 input_size:int = 28*28, 
                 num_hidden:int = 100,
                 pyc_ratio:float = 0.8,
                 context_classes:int = 10,
                 context_dim:int = 32,
                 activation_threshold:float = 0.0):
        super().__init__()
        self.n_neurons = num_hidden
        self.pyc_size = int(self.n_neurons * pyc_ratio)
        self.pv_size = self.n_neurons - self.pyc_size
        # context embedding
        self.context = nn.Embedding(num_embeddings=context_classes, embedding_dim=context_dim)

        # RNN pool with pyc's and pv's; each can have their own time constant
        self.neurons = EMA(shape=(1, self.n_neurons), alpha=0.1, learnable_alpha=True, matrix_alpha=True)
        
        # Randomly assign neurons to be PV = 0 (20 %) or PyC = 1 (80 %)
        self.pyc_indices = torch.multinomial(torch.tensor([1 - pyc_ratio, pyc_ratio]), 
                                             num_samples=self.n_neurons, replacement=True).bool()
        self.pv_indices = ~self.pyc_indices
        
        # Initialize weights
        self.w_ff = nn.Linear(input_size, self.n_neurons, bias=False)
        self.w_fb = nn.Linear(context_dim, self.pyc_size, bias=False)
        self.w_lat = nn.Linear(self.n_neurons, self.n_neurons, bias=False)
        self.w_readout = nn.Linear(self.n_neurons, input_size, bias=True)

        # Activation function, possible to raise threshold to prevent subthreshold responses
        self.activation = ThresholdReLU(threshold=activation_threshold)
    
    def forward_step(self, x:torch.Tensor, context_labels:torch.Tensor) -> torch.Tensor:
        # Enforce Dale's law at every step
        self._dales_law()
        Y_t = self.neurons.ema
        # "External" excitatory drives from other areas
        FF_drive = self.w_ff(x) # to all neurons
        # Feedback from higher area carrying invariant contextual information
        FB_drive = self.w_fb(self.context(context_labels)) # to PyCs only
        # Recurrent lateral dynamics based on previous time-step
        Lat_drive = self.w_lat(Y_t)
        
        # Common drives to PyCs and PVs
        drive = FF_drive + Lat_drive
        # Add FB drive to PyCs only
        drive[:, self.pyc_indices] += FB_drive 

        Y_next = self.neurons(self.activation(drive))

        return x, Y_t, Y_next, 

    def forward(self, x:torch.Tensor, context_labels:torch.Tensor, n_steps:int = 10) -> torch.Tensor:
        # Assume input has a temporal dimension
        for _ in range(n_steps):
            x = self.forward_step(x, context_labels)
        return x

    @torch.no_grad()
    def _dales_law(self, mode:Literal['clamp', 'exp'] = 'clamp'):
        # Enforce Dale's law by clamping weights to be non-negative or non-positive as appropriate
        # Excitatory FF and FB input
        self.w_ff.weight = nonnegative(self.w_ff.weight)
        self.w_fb.weight = nonnegative(self.w_fb.weight)
        # PV->PV is inhibitory
        self.w_lat.weight[self.pv_indices, self.pv_indices] = nonpositive(self.w_lat.weight[self.pv_indices, self.pv_indices])
        # PV->PyC is inhibitory
        self.w_lat.weight[self.pyc_indices, self.pv_indices] = nonpositive(self.w_lat.weight[self.pyc_indices, self.pv_indices])
        # PyC->PV is excitatory
        self.w_lat.weight[self.pv_indices, self.pyc_indices] = nonnegative(self.w_lat.weight[self.pv_indices, self.pyc_indices])
        # PyC -> PyC is excitatory
        self.w_lat.weight[self.pyc_indices, self.pyc_indices] = nonnegative(self.w_lat.weight[self.pyc_indices, self.pyc_indices])