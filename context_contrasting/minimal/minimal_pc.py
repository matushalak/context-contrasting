# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
from typing import Literal

from context_contrasting.utils import EMA, ThresholdReLU, nonnegative, randn_reparam

# TODO
class PCNeuron(nn.Module):
    """
    Minimal predictive-coding model with:
      - one pyramidal neuron y (scalar),
      - two PV neurons p (vector of size 2),
      - feedforward input x (size 2),
      - contextual input c (size 2).


    """
    # using randn_reparam
    def __init__(
        self,
        n_features: int = 2,
        n_pv: int = 2,
        n_context: int = 2,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        pass