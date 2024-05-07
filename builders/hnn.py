import torch
import numpy as np
from scipy.integrate import solve_ivp

class HNN(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(HNN, self).__init__()

        self.model = model

    def forward(self, x):
        H = self.model(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        q_dot = dH[:, 1]
        p_dot = -dH[:, 0]

        dH[:, 0] = q_dot
        dH[:, 1] = p_dot

        return dH
    
def build(
    model: torch.nn.Module
):
    return HNN(model)
    
    