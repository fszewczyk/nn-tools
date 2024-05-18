from torch import nn 
from nn_tools.shared.constants import *

def build(input_dim: int,
        hidden_layers: list,
        out_dim: int,
        activations='relu'):
    if not isinstance(activations, list):
        activations = [activations] * (len(hidden_layers))
        activations.append('identity')
    elif len(activations) != len(hidden_layers) + 1:
        print(f'To build an MLP, you need to pass the same amount of activations as layers. You want to create {len(hidden_layers) + 1} layers, while providing {len(activations)} activation functions.')
        return None
    
    layer_sizes = [input_dim, *hidden_layers, out_dim]

    layers = []

    # Iterate over the layer sizes and create linear layers
    for i in range(1, len(layer_sizes)):
        in_size = layer_sizes[i - 1]
        out_size = layer_sizes[i]
        layer = nn.Linear(in_size, out_size)

        # Append the linear layer to the list
        layers.append(layer)

        # Add activation after all but the last layer
        if not activations[i - 1] in ACTIVATION_FUNCTIONS.keys():
            print(f"{activations[i - 1]} is not a valid activation function. Should be one of {ACTIVATION_FUNCTIONS.keys()}. Using ReLU as default.")
            activations[i - 1] = 'relu'
        
        activation_fun = ACTIVATION_FUNCTIONS[activations[i - 1]]
        if activations[i - 1] == 'snake':
            layers.append(activation_fun(out_size, 1.0))
        else:
            layers.append(activation_fun())

    # Combine all layers into a Sequential model
    mlp = nn.Sequential(*layers)

    return mlp