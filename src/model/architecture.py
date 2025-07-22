# src/model/architecture.py

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def create_cnn_from_config(config: Dict[str, Any], input_shape: Tuple[int, int, int]) -> nn.Module:
    """Dynamically builds a PyTorch CNN model from a configuration dictionary.

    This function reads the model architecture defined in the `model.architecture`
    section of the configuration file. It constructs a sequential nn.Module by
    iterating through the layer definitions, allowing for flexible network
    design without changing the source code. It also initializes the model's
    lazy layers by performing a dummy forward pass.

    Parameters
    ----------
    config : Dict[str, Any]
        The project's main configuration dictionary.
    input_shape : Tuple[int, int, int]
        The shape of a single input sample in (C, H, W) format.

    Returns
    -------
    nn.Module
        The constructed and initialized PyTorch model.

    Raises
    ------
    ValueError
        If the configuration specifies a layer type that is not supported.
    """
    architecture_config = config['model']['architecture']
    
    class DynamicCNN(nn.Module):
        """A sequential CNN model whose architecture is defined at runtime.

        This class dynamically assembles a list of PyTorch layers
        (nn.ModuleList) based on a provided configuration. It is designed to be
        instantiated within the `create_cnn_from_config` function.
        """
        def __init__(self):
            """Initializes and builds the layers of the neural network.

            Iterates through the architecture definition, adding PyTorch modules
            to the `self.layers` list and tracking the number of channels for
            convolutional and normalization layers.
            """
            super().__init__()
            self.layers = nn.ModuleList()
            in_channels = input_shape[0]

            for layer_conf in architecture_config:
                layer_type = layer_conf['layer']
                params = layer_conf.get('params', {})

                if layer_type == 'Conv2D':
                    self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=params['filters'], 
                                                 kernel_size=tuple(params['kernel_size']), stride=tuple(params.get('strides', (1,1)))))
                    in_channels = params['filters']
                elif layer_type == 'LeakyReLU':
                    self.layers.append(nn.LeakyReLU(negative_slope=params.get('alpha', 0.01)))
                elif layer_type == 'ReLU':
                    self.layers.append(nn.ReLU())
                elif layer_type == 'BatchNormalization':
                    self.layers.append(nn.BatchNorm2d(num_features=in_channels))
                elif layer_type == 'MaxPool2D':
                    self.layers.append(nn.MaxPool2d(kernel_size=tuple(params['pool_size'])))
                elif layer_type == 'Flatten':
                    self.layers.append(nn.Flatten())
                elif layer_type == 'Dropout':
                    self.layers.append(nn.Dropout(p=params.get('rate', 0.5)))
                elif layer_type == 'Dense':
                    self.layers.append(nn.LazyLinear(out_features=params['units']))
                elif layer_type == 'Sigmoid':
                    self.layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unbekannter Layer-Typ: {layer_type}")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Defines the forward pass of the model.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor.

            Returns
            -------
            torch.Tensor
                The output tensor of the network.
            """
            for layer in self.layers:
                x = layer(x)
            return x
            
    print("\n--- Building CNN model from configuration ---")
    model = DynamicCNN()
    dummy_input = torch.randn(1, *input_shape)
    model(dummy_input)
    print("✅ Model created successfully.")
    
    return model