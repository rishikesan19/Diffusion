import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler, DDPMScheduler
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusion_models.models.conditional.attribute_embedder import AttributeEmbedder


class AttributeDiffusionPipeline(DiffusionPipeline):
    """
    Custom diffusion pipeline for generating images conditioned on 40 binary attributes
    and optionally part-wise segmentation masks.
    Supports both DDPM and DDIM schedulers for sampling.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        attribute_embedder: AttributeEmbedder,
        image_size: int = 256
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            attribute_embedder=attribute_embedder
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_size = image_size

        expected_sample_size = image_size // self.vae_scale_factor
        if self.unet.config.sample_size != expected_sample_size:
            raise ValueError(
                f"UNet sample_size ({self.unet.config.sample_size}) != expected ({expected_sample_size})"
            )

    @torch.no_grad()
    def __call__(
        self,
        attributes: Optional[torch.Tensor] = None,
        segmentation: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        decode_batch_size: int = 2,
        eta: float = 0.0,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:

        if attributes is None:
            raise ValueError("`attributes` must be provided for conditional generation.")
        if batch_size is None:
            batch_size = attributes.size(0)
        if attributes.size(1) != 40:
            raise ValueError("Attributes tensor must have shape (batch_size, 40)")

        device = self.unet.device
        dtype = self.unet.dtype
        attributes = attributes.to(device, dtype)
        if segmentation is not None:
            segmentation = segmentation.to(device)

        latent_size = self.unet.config.sample_size
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_size, latent_size),
            device=device,
            dtype=dtype,
            generator=generator
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma

        encoder_hidden_states = self.attribute_embedder(attributes)

        with tqdm(total=len(timesteps), desc="Sampling") as pbar:
            for t in timesteps:
                t = t.to(device)

                if segmentation is not None:
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        segmentation=segmentation
                    ).sample
                else:
                    noise_pred = self.unet(
                        latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample

                if isinstance(self.scheduler, DDIMScheduler):
                    step_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        eta=eta,
                        use_clipped_model_output=False,
                        generator=generator,
                    )
                else:
                    step_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        generator=generator,
                    )

                latents = step_output.prev_sample
                del noise_pred, step_output
                torch.cuda.empty_cache()
                pbar.update(1)

        latents = latents / self.vae.config.scaling_factor
        all_images = []
        target_size = (self.image_size, self.image_size)

        for i in tqdm(range(0, batch_size, decode_batch_size), desc="Decoding"):
            batch_latents = latents[i:i + decode_batch_size]
            torch.cuda.empty_cache()

            batch_images = self.vae.decode(batch_latents).sample
            batch_images = (batch_images / 2 + 0.5).clamp(0, 1)

            if output_type == "pil":
                for img in batch_images:
                    img_np = img.cpu().numpy().transpose(1, 2, 0) * 255
                    img_pil = Image.fromarray(img_np.astype(np.uint8))
                    if img_pil.size != target_size:
                        img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
                    all_images.append(img_pil)
            else:
                if batch_images.shape[-2:] != target_size:
                    batch_images = torch.nn.functional.interpolate(
                        batch_images, size=target_size, mode='bicubic', align_corners=False
                    )
                    batch_images = batch_images.clamp(0, 1)
                all_images.append(batch_images.cpu())

            del batch_images, batch_latents
            torch.cuda.empty_cache()

        if output_type != "pil":
            all_images = torch.cat(all_images, dim=0)

        return {"sample": all_images} if return_dict else all_images
