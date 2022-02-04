import torch
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
        self.weight = torch.nn.Parameter(
            torch.empty((self.input_dim, self.embed_dim))
        )

        # The embedding bias term, which is often the phase offset of the period activation function.
        self.bias = torch.nn.Parameter(
            torch.empty(self.embed_dim)
        )

        # The periodic activation function.
        self.act_func = act_func

        # Call parameter reset.
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.uniform_(self.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Perform affine transformation (y = xw + b) on the input w.r.t. using the module weight and bias.
        affine = torch.matmul(x, self.weight) + self.bias

        # Pass the 1<=i<=k parts of the affine transformation through the activation function.
        # Note that the i=0 index is unchanged.
        affine[...,1:] = self.act_func(affine[...,1:])

        # Ensure that the output is 3-dimensional.
        out = affine.view(affine.size(0), affine.size(1), -1)
        return out




        # # Separate the first element from the affine transformation.
        # # From the paper, these are indexes i=0 and 1<=i<=k.
        # affine_0, affine_k = torch.split(
        #     affine,
        #     [1, self.embed_dim-1],
        #     dim=-1,
        # )

        # # Pass the 1<=i<=k parts of the affine transformation through the activation function.
        # affine_k = self.act_func(affine_k)

        # # Rebuild the tensor with both i=0 and 1<=i<=k components.
        # y = torch.cat((affine_0, affine_k), dim=-1)