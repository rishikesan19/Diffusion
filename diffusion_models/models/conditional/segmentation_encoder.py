import torch
import torch.nn as nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

class SegmentationEncoder(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device="cuda"):
        super().__init__()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.encoder = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
        self.device = device

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, segmentation_image: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(images=[img.cpu().permute(1, 2, 0).numpy()
                                                for img in segmentation_image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        return outputs.hidden_states[-1]  # (B, C, H', W')
