from torch import nn

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': nn.Identity
}

LOSS_FUNCTIONS = {
    'L2': nn.MSELoss,
    'L1': nn.L1Loss,
    'CrossEntropy': nn.CrossEntropyLoss
}
