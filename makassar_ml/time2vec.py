import torch
import torch.nn
from typing import Callable, TypeVar

# Type definition for activation function.
T = TypeVar('T')
ActivationFunction = Callable[[T], T]

class Time2Vec(torch.nn.Module):
    """Implementation of the Time2Vec embedding technique as proposed by Kazemi et al. 2019.
    https://arxiv.org/abs/1907.05321
    """
    def __init__(self, 
        input_dim: int,
        embed_dim: int = 512,
        act_func: ActivationFunction[torch.Tensor] = torch.sin,
        ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # The embedding weight, which is often the frequency of the period activation function.
        self.weight = torch.nn.parameter.Parameter(
            torch.randn(self.input_dim, self.embed_dim)
        )

        # The embedding bias term, which is often the phase offset of the period activation function.
        self.bias = torch.nn.parameter.Parameter(
            torch.randn(self.embed_dim)
        )

        # The periodic activation function.
        self.act_func = act_func