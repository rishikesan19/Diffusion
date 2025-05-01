"""Latent Conditional UNet model for attribute-based latent diffusion."""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DConditionModel:
    """Create and return the Conditional UNet2D model.
    
    This model is designed for latent diffusion, operating in the VAE latent space
    and conditioned on attribute vectors. The architecture is memory-efficient while
    maintaining strong conditioning through cross-attention at key resolutions.
    
    Args:
        config: Training configuration object
        
    Returns:
        UNet2DConditionModel
    """
    # Create the UNet model for latent diffusion
    model = UNet2DConditionModel(
        # Latent space parameters
        sample_size=32,  # 32x32 latents (256/8 due to VAE downsampling)
        in_channels=4,    # VAE latent space channels
        out_channels=4,   # Noise prediction in latent space
        
        # Downsampling blocks with selective attention
        down_block_types=(
            "CrossAttnDownBlock2D",    # 32x32 -> 16x16 with cross-attention
            "CrossAttnDownBlock2D",    # 16x16 -> 8x8 with cross-attention
            "DownBlock2D",             # 8x8 -> 4x4 standard downsampling
            "DownBlock2D",             # 4x4 -> 2x2 standard downsampling
        ),
        
        # Upsampling blocks with symmetric attention
        up_block_types=(
            "UpBlock2D",               # 2x2 -> 4x4 standard upsampling
            "UpBlock2D",               # 4x4 -> 8x8 standard upsampling
            "CrossAttnUpBlock2D",      # 8x8 -> 16x16 with cross-attention
            "CrossAttnUpBlock2D",      # 16x16 -> 32x32 with cross-attention
        ),
        
        # Architecture parameters
        block_out_channels=(128, 256, 512, 512),        # Channel dimensions per block
        layers_per_block=2,                             # Two ResNet layers per block for better capacity
        cross_attention_dim=config.cross_attention_dim, # Dimension of cross-attention features
        attention_head_dim=8,                           # Size of attention heads
        
        # Model configuration
        use_linear_projection=True,               # Memory-efficient attention
        num_class_embeds=None,                    # No class conditioning
        only_cross_attention=False,               # Enable both self and cross attention
        
        # Architecture details
        act_fn="silu",                           # SiLU activation function
        norm_num_groups=32,                       # Group normalization
        norm_eps=1e-5,                           # Numerical stability
        cross_attention_norm="layer_norm",        # Cross-attention normalization
    )
    
    if hasattr(config, "device"):
        model = model.to(config.device)
        
    print(f"\nCreated UNet2DConditionModel with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model 