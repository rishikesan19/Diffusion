import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class SegmentationEncoder(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device="cuda"):
        super().__init__()
        self.encoder = SegformerForSemanticSegmentation.from_pretrained(model_name).base_model.to(device)
        self.device = device

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, segmentation_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segmentation_tensor: Float tensor (B, 3, H, W) — RGB preprocessed segmentation mask

        Returns:
            features: Last hidden state features (B, C)
        """
        with torch.no_grad():
            outputs = self.encoder(segmentation_tensor.to(self.device)).last_hidden_state
            pooled = torch.mean(outputs, dim=[2, 3])  # Global average pool
        return pooled  # Shape: (B, hidden_dim)
