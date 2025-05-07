"""Training loop implementation for diffusion models."""

import os
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, VQModel
import wandb

from diffusion_models.utils.generation import generate_grid_images, generate_grid_images_attributes
from diffusion_models.utils.metrics import generate_and_calculate_fid, generate_and_calculate_fid_attributes, generate_and_calculate_fid_attr_seg
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.losses.info_nce import info_nce

def train_loop(
    config, 
    model, 
    noise_scheduler, 
    optimizer, 
    train_dataloader, 
    lr_scheduler=None, 
    val_dataloader=None, 
    preprocess=None,
    is_conditional=False,
    grid_attributes=None,
    val_attributes=None,
    attribute_embedder=None,
    vae=None,
    ema=None
):
    """Main training loop.
    
    Args:
        config: Training configuration
        model: The UNet model to train
        noise_scheduler: The noise scheduler
        optimizer: The optimizer
        train_dataloader: Training data loader
        lr_scheduler: Learning rate scheduler
        val_dataloader: Optional validation data loader
        preprocess: Optional preprocessing transform
        is_conditional: Whether the model is conditional on attributes
        grid_attributes: Optional tensor of attributes for grid visualization
        val_attributes: Optional tensor of attributes for FID validation
        attribute_embedder: Optional module to project attributes to hidden states
        vae: Optional VAE model
        ema: Optional EMA model
    """

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "optimizer"), exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "best_model"), exist_ok=True)
        accelerator.init_trackers("train_example")

    # Initialize best FID score tracking
    best_fid_score = float('inf')
    best_epoch = 0

    # Load EMA weights if resuming training
    if ema:
        ema_path = os.path.join(config.output_dir, "ema.pt")
        if os.path.exists(ema_path):
            print(f"[INFO] Loading EMA from {ema_path}")
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))

    # Prepare model, optimizer, dataloaders with accelerator
    prepare_args = [model, optimizer, train_dataloader, lr_scheduler]
    if is_conditional and attribute_embedder is not None:
        prepare_args.append(attribute_embedder)

    prepared = accelerator.prepare(*prepare_args)
    if is_conditional and attribute_embedder is not None:
        model, optimizer, train_dataloader, lr_scheduler, attribute_embedder = prepared
    else:
        model, optimizer, train_dataloader, lr_scheduler = prepared

    if vae is not None:
        vae = accelerator.prepare(vae)

    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    if ema:
        ema.to(accelerator.device)

    global_step = 0

    # Training loop
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if vae is not None:
            vae.eval()  # VAE is pretrained, no training needed

        for step, batch in enumerate(train_dataloader):
            # Handle both conditional and unconditional cases
            if is_conditional:
                if isinstance(batch, dict):
                    clean_images = batch["image"]
                    attributes = batch["attribute"]
                    segmentations = batch.get("segmentation")
                else:
                    clean_images, attributes = batch
                    segmentations = None
            else:
                clean_images = batch["images"]

            # Sample noise to add to the images
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            if vae is not None:
                # For conditional model - encode images to latent space
                with torch.no_grad():
                    if isinstance(vae, VQModel):
                        # VQ-VAE
                        latents = vae.encode(clean_images).latents  # (batch_size, 4, 32, 32)
                    else:
                        # AutoencoderKL
                        latents = vae.encode(clean_images).latent_dist.sample()   # (batch_size, 4, 32, 32)
                    latents = latents * vae.config.scaling_factor
                latents = latents.to(clean_images.device)
                noise = torch.randn_like(latents).to(latents.device)
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
            else:
                # For unconditional model - use the original images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if is_conditional and attribute_embedder is not None:
                    # Project attributes to hidden states
                    encoder_hidden_states = attribute_embedder(attributes)
                    noise_pred = model(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        segmentation=segmentations.to(noisy_images.device) if segmentations is not None else None,
                        return_dict=False
                    )[0]
                else:
                    # For unconditional model
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                # Calculate loss
                if is_conditional and config.use_embedding_loss:
                    # Squeeze the sequence dimension from encoder_hidden_states
                    encoder_hidden_states = encoder_hidden_states.squeeze(1)  # Shape: (batch_size, hidden_dim)
                    # print(f"Using embedding loss")
                    # print(f"encoder_hidden_states: {encoder_hidden_states.shape}")
                    # print(f"attributes: {attributes.shape}")
                    embedding_loss = info_nce(encoder_hidden_states, attributes)
                    diffusion_loss = F.mse_loss(noise_pred, noise)
                    # Loss = diffusion loss + lambda * embedding loss
                    loss = diffusion_loss + config.embedding_loss_lambda * embedding_loss
                    # print(f"embedding_loss: {embedding_loss.item()}")
                    # print(f"diffusion_loss: {diffusion_loss.item()}")
                    # print(f"loss: {loss.item()}")
                else:
                    # Loss = diffusion loss
                    loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema:
                    ema.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if config.use_wandb:
                wandb.log({
                    "train/batch_loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch
                }, step=global_step)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # Generate and save sample images
                if is_conditional:
                    # Create a pipeline for visualization
                    if vae is not None:
                        # Create conditional pipeline with VAE for latent conditioning
                        pipeline = AttributeDiffusionPipeline(
                            unet=accelerator.unwrap_model(model),
                            vae=vae,
                            scheduler=noise_scheduler,
                            attribute_embedder=attribute_embedder,
                            image_size=config.image_size
                        )
                    else:
                        # Conditional pipeline with direct pixel-space
                        raise NotImplementedError("Pixel-space conditional generation not supported yet")

                    # Move grid attributes to correct device
                    grid_attributes = grid_attributes.to(accelerator.device)
                    _, image_grid = generate_grid_images_attributes(
                        config, epoch, pipeline,
                        attributes=grid_attributes
                    )
                else:
                    # For unconditional generation, use DDPMPipeline
                    if ema:
                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(ema.ema_model), scheduler=noise_scheduler)
                    else:
                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                    # Move the pipeline to the accelerator device.
                    pipeline = pipeline.to(accelerator.device)

                    # For unconditional model
                    _, image_grid = generate_grid_images(config, epoch, pipeline)

                # Log grid images to WandB
                if config.use_wandb:
                    wandb.log({
                        "validation/grid_images": wandb.Image(image_grid),
                        "validation/epoch": epoch
                    })

                # Calculate FID if validation dataset is available
                if val_dataloader:
                    print(f"Calculating FID score at epoch {epoch + 1}...")

                    if is_conditional and val_attributes is not None:
                        # Move validation attributes to correct device
                        val_attributes = val_attributes.to(accelerator.device)
                        if config.use_segmentation:
                            # Compute FID for attribute+segmentation case
                            fid_score = generate_and_calculate_fid_attr_seg(
                                pipeline=pipeline,
                                val_dataloader=val_dataloader,
                                device=accelerator.device,
                                preprocess=preprocess,
                                num_train_timesteps=config.num_train_timesteps,
                                num_samples=config.val_n_samples,
                                attributes=val_attributes
                            )
                        else:
                            # Use attribute-specific FID calculation
                            fid_score = generate_and_calculate_fid_attributes(
                                pipeline=pipeline,
                                val_dataloader=val_dataloader,
                                device=accelerator.device,
                                preprocess=preprocess,
                                num_train_timesteps=config.num_train_timesteps,
                                num_samples=config.val_n_samples,
                                attributes=val_attributes
                            )
                    else:
                        # Use standard FID calculation for unconditional model
                        fid_score = generate_and_calculate_fid(
                            pipeline=pipeline,
                            val_dataloader=val_dataloader,
                            device=accelerator.device,
                            preprocess=preprocess,
                            num_train_timesteps=config.num_train_timesteps,
                            num_samples=config.val_n_samples
                        )

                    print(f"FID Score: {fid_score:.2f}")

                    # Log to WandB
                    if config.use_wandb:
                        wandb.log({"validation/fid_score": fid_score})

                    # Save best model if FID score improves
                    if fid_score < best_fid_score:
                        best_fid_score = fid_score
                        best_epoch = epoch
                        print(f"New best FID score: {best_fid_score:.2f} at epoch {best_epoch}")
                        # Save best model
                        pipeline.save_pretrained(os.path.join(config.output_dir, "best_model"))
                        os.makedirs(os.path.join(config.output_dir, "best_model", "optimizer"), exist_ok=True)
                        # Save optimizer state for best model
                        torch.save({
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                            "loss": loss.item(),
                            "fid_score": best_fid_score,
                        }, os.path.join(config.output_dir, "best_model", "optimizer", "optimizer.pth"))
                        if ema:
                            torch.save(ema.state_dict(), os.path.join(config.output_dir, "best_model", "ema.pt"))

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # Save most recent checkpoint
                pipeline.save_pretrained(config.output_dir)
                # Save optimizer and scaler for resuming
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                    "loss": loss.item(),
                }, os.path.join(config.output_dir, "optimizer", "optimizer.pth"))
                if ema:
                    torch.save(ema.state_dict(), os.path.join(config.output_dir, "ema.pt"))
