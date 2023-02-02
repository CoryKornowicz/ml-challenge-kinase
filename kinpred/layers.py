
import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import List, Tuple, Optional

# torch layer to map each element to 0 or 1 that is differentiable
class BinaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# Module to implement the function above
class Binary(nn.Module):
    def forward(self, input):
        return BinaryFunction.apply(input)
    


def get_ff_layer_sizes(in_size: int, out_size: int, hidden_layers: int) -> List[Tuple[int, int]]:
    """Generate cascading layer sizes for a feedforward network.
    ex: in_size = 100, out_size = 10, hidden_layers = 3
    returns [(100, 55), (55, 32), (32, 21), (21, 10)]
    Args:
        in_size (int): input dimension
        out_size (int): outout dimension
        hidden_layers (int): number of intermediate hidden layers

    Returns:
        List[Tuple[int, int]]: list of (input,output) pairs representing each layer size in the network
    """
    sizes = [in_size]
    sizes.extend((sizes[-1] + out_size)//2 for _ in range(hidden_layers))
    sizes.append(out_size)
    return [(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]


def linear_block(in_size: int, out_size: int, act: bool = False, act_func: nn.Module = None, dropout: Optional[float] = 0.15) -> nn.Sequential:
    """Linear layer block of linear -> batchnorm -> dropout -> optional(activation). Automates the construction of middle layers to the feedforward network.

    Args:
        in_size (int): input dimension
        out_size (int): output dimension
        act (bool, optional): Apply the activation function or not. Defaults to False.
        act_func (nn.Module, optional): Which activation function to apply. Defaults to None.
        dropout (float, optional): Dropout rate to apply. Defaults to 0.15.

    Returns:
        nn.Sequential: returns a linear layer block for the feedforward network
    """
    layers = [nn.Linear(in_size, out_size)]
    layers.append(nn.BatchNorm1d(out_size))
    if dropout:
        layers.append(nn.Dropout(dropout))
    if act:
        layers.append(act_func)
    return nn.Sequential(*layers)


# Feedforward network with variable number of hidden layers, activation function, and dropout
class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers, act = nn.SiLU(), dropout = 0.15):
        super(MLPBlock, self).__init__()
        
        layer_sizes = get_ff_layer_sizes(in_channels, out_channels, layers)
        layernet = [linear_block(inp, outp, True, act_func=act, dropout=dropout) for (inp, outp) in layer_sizes[:-1]]
        layernet.append(linear_block(layer_sizes[-1][0], layer_sizes[-1][1], False, None, dropout=None))
        self.net = nn.Sequential(*layernet)
        
    def forward(self, x):
        # Note: does not apply a final activation function
        return self.net(x)