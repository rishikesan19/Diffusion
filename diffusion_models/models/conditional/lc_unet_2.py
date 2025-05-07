import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import SegformerForSemanticSegmentation
from diffusion_models.config import TrainingConfig


class SegmentationConditionedUNet(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()

        # Load segmentation encoder if checkpoint provided
        if hasattr(config, "segmentation_encoder_checkpoint") and config.segmentation_encoder_checkpoint:
            self.segmentation_encoder = SegformerForSemanticSegmentation.from_pretrained(
                config.segmentation_encoder_checkpoint
            ).base_model
            for param in self.segmentation_encoder.parameters():
                param.requires_grad = False
            self.segmentation_encoder.eval()
            self.use_segmentation = True
            self.segmentation_output_dim = self.segmentation_encoder.config.hidden_sizes[-1]
            self.segmentation_proj = nn.Linear(self.segmentation_output_dim, config.cross_attention_dim)
        else:
            self.segmentation_encoder = None
            self.use_segmentation = False

        # Calculate sample_size based on image_size and VAE/VQ-VAE downsampling
        sample_size = config.image_size // 4

        # Create UNet2DConditionModel for latent diffusion
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            cross_attention_dim=config.cross_attention_dim,  # Must match attribute embedder output dim
            attention_head_dim=8,
            use_linear_projection=True,
            num_class_embeds=None,
            only_cross_attention=False,
            act_fn="gelu",
            norm_num_groups=32,
            norm_eps=1e-5,
            cross_attention_norm="layer_norm",
        )

        if hasattr(config, "device"):
            self.unet = self.unet.to(config.device)
            if self.segmentation_encoder:
                self.segmentation_encoder = self.segmentation_encoder.to(config.device)

        # Log model stats
        param_count = sum(p.numel() for p in self.unet.parameters())
        batch_size = 16
        latent_size = sample_size * sample_size * 3
        memory_per_sample = param_count * 4
        total_memory = memory_per_sample * batch_size

        print(f"\nCreated UNet2DConditionModel:")
        print(f"Parameters: {param_count:,}")
        print(f"Sample size: {sample_size}x{sample_size} (for {config.image_size}x{config.image_size} images)")
        print(f"Approximate memory usage: {total_memory / (1024**3):.2f} GB for batch_size={batch_size}")

        self.config = self.unet.config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x, timestep, encoder_hidden_states=None, segmentation=None, **kwargs):
        # Handle segmentation-based conditioning
        if self.use_segmentation and segmentation is not None:
            if segmentation.shape[1] == 1:
                segmentation = segmentation.repeat(1, 3, 1, 1)
            with torch.no_grad():
                seg_features = self.segmentation_encoder(segmentation).last_hidden_state
                seg_features = torch.mean(seg_features, dim=[2, 3])  # [B, C]
            seg_features = self.segmentation_proj(seg_features)      # [B, D]
            seg_features = seg_features.unsqueeze(1)                # [B, 1, D]
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.cat([encoder_hidden_states, seg_features], dim=1)
            else:
                encoder_hidden_states = seg_features

        return self.unet(x, timestep, encoder_hidden_states=encoder_hidden_states)


def create_model(config: TrainingConfig) -> SegmentationConditionedUNet:
    """Factory function to create the SegmentationConditionedUNet model."""
    return SegmentationConditionedUNet(config)
