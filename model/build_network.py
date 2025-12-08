#!/usr/bin/env python3

import yaml
import sys
import torch

# import lion_pytorch  # Importing Lion Optimizer (Meta AI)

string_to_torch = {
    # Layers
    "GRU": torch.nn.GRU,
    "LSTM": torch.nn.LSTM,
    "DENSE": torch.nn.Linear,
    "DROPOUT": torch.nn.Dropout(p=0.1),  # Added Dropout layer
    "LAYERNORM": torch.nn.LayerNorm,  # LayerNorm for GRU stability
    "BATCHNORM": torch.nn.BatchNorm1d,  # BatchNorm for stable training
    "GROUPNORM": torch.nn.GroupNorm,  # GroupNorm for smaller batch sizes
    
    # Activations
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,  # Prevents dead neurons
    "Mish": torch.nn.Mish,
    "Softplus": torch.nn.Softplus,
    "Sigmoid": torch.nn.Sigmoid,
    "Softmax": torch.nn.Softmax,
    "ELU": torch.nn.ELU,
    "SiLU": torch.nn.SiLU,  # Swish activation
    # "SiLU": torch.nn.ReLU6,  # Swish activation
    "ReLU6": torch.nn.ReLU6,  # Swish activation
    "Tanh": torch.nn.Tanh,  # Added Tanh activation
    "GELU": torch.nn.GELU,  # More stable activation for deep networks
    
    # Loss Functions
    "MSE": torch.nn.MSELoss,  
    "MAE": torch.nn.SmoothL1Loss,  
    "L1": torch.nn.L1Loss,  
    "Huber": torch.nn.HuberLoss,  # Huber loss for robustness
    
    # Optimizers
    "Adam": torch.optim.Adam,
    "NAdam": torch.optim.NAdam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,  
    "RMSprop": torch.optim.RMSprop,  
    # "Lion": lion_pytorch.Lion,  # Meta AI's new optimizer for precision learning
    "Adadelta": torch.optim.Adadelta,  # Robust optimizer for adaptive learning rates
    
    # Learning Rate Schedulers
    "StepLR": torch.optim.lr_scheduler.StepLR,  
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,  # One-cycle policy for better convergence
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,  # Adaptive LR scaling
    "CosineDecay" : torch.optim.lr_scheduler.CosineAnnealingLR, 
    "CosineWarmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,  # Cosine annealing with warm restarts
}


def build_network(param_dict):
    horizon = param_dict["MODEL"]["HORIZON"]
    num_states = len(param_dict["STATE"])
    num_actions = len(param_dict["ACTIONS"])
    layers = []
    
    for i in range(len(param_dict["MODEL"]["LAYERS"])):
        layer_dict = param_dict["MODEL"]["LAYERS"][i]
        layer_type = list(layer_dict.keys())[0]  # Get the first key (e.g., "DENSE", "GROUPNORM")
        
        if i == 0:
            input_size = (num_states + num_actions) * horizon
        else:
            input_size = param_dict["MODEL"]["LAYERS"][i-1].get("OUT_FEATURES", input_size)  # Use previous output size
        
        # Handle layers that do not require OUT_FEATURES
        if "OUT_FEATURES" in layer_dict:
            output_size = layer_dict["OUT_FEATURES"]
        else:
            output_size = input_size  # For GroupNorm, BatchNorm, etc., keep input_size same
        
        module = create_module(
            layer_type, 
            input_size, 
            horizon, 
            output_size, 
            layer_dict.get("LAYERS"), 
            layer_dict.get("ACTIVATION")
        )
        
        layers += module
    
    return layers



def create_module(name, input_size, horizon, output_size, layers=None, activation=None):
    if layers:
        module = [string_to_torch[name](input_size // horizon, horizon, layers, batch_first=True)]
    elif activation:
        module = [string_to_torch[name](input_size, output_size), string_to_torch[activation]()]
    else:
        module = [string_to_torch[name](input_size, output_size)]
    return module
