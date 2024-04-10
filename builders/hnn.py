import torch
import numpy as np
from scipy.integrate import solve_ivp

class HNN(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_dim: int):
        super(HNN, self).__init__()

        self.model = model
        self.M = self.permutation_tensor(input_dim)

    def forward(self, x):
        H = self.model(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        q_dot = dH[:, 1]
        p_dot = -dH[:, 0]

        dH[:, 0] = q_dot
        dH[:, 1] = p_dot

        return dH

    def permutation_tensor(self, n):
        M = torch.eye(n)
        M = torch.cat([M[n//2:], -M[:n//2]])

        return M

    def integrate(self, y0, t_span=[0, 5], **kwargs):
        def fun(t, np_x):
            x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
            x = x.view(1, np.size(np_x))
            dx = self.time_derivative(x).data.numpy()
            return dx
        return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    
def build(
    model: torch.nn.Module,
    input_dim: int = 3
):
    return HNN(model, input_dim)
    
    