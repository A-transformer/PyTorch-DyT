import torch
import torch.nn as nn
from timm.layers import LayerNorm2d

# Custom Dynamic Tanh Layer
class DynamicTanh(nn.Module):
    """
    Dynamic Tanh layer that adjusts the scaling factor alpha dynamically based on the input.

    Args:
        normalized_shape (int or tuple): The shape to apply normalization.
        channels_last (bool): Whether to use channel-last format (NHWC).
        alpha_init_value (float, optional): Initial value of alpha, default is 0.5.
    """
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)  # Learnable scaling factor
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # Learnable weight parameter
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # Learnable bias parameter

    def forward(self, x):
        """ Forward pass """
        x = torch.tanh(self.alpha * x)  # Apply scaled tanh
        if self.channels_last:
            x = x * self.weight + self.bias  # Apply weight and bias for channel-last format
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]  # Apply weight and bias for channel-first
        return x

    def extra_repr(self):
        """ Extra representation for debugging """
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

# Function to convert LayerNorm layers to DynamicTanh

def convert_ln_to_dyt(module):
    """
    Converts LayerNorm layers in a model to DynamicTanh layers.

    Args:
        module (nn.Module): Input PyTorch module.

    Returns:
        nn.Module: Converted PyTorch module.
    """
    module_output = module

    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))

    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    
    del module  # Delete the original module
    return module_output


if __name__ == "__main__":
    # Example model with LayerNorm layers
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.LayerNorm(10),
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        LayerNorm2d(16),  # Corrected usage
        nn.ReLU()
    )

    # Convert LayerNorm layers to DynamicTanh
    converted_model = convert_ln_to_dyt(model)

    # Print the modified model structure
    print(converted_model)

