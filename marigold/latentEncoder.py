from diffusers import AutoencoderKL
import torch
from torch import nn


class LatentEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str = "stabilityai/stable-diffusion-2-base",
        device: str = "cuda",
        deterministic: bool = False
    ):
        """
        VAE encoder for Marigold depth estimation

        Args:
            pretrained_model_path: Path to pretrained Stable Diffusion model
            device: Device to run model on ('cuda' or 'cpu')
            deterministic: If True, use mode instead of sampling (for reproducibility)
        """
        super().__init__()
        self._pretrained_model_path = pretrained_model_path
        self._vae = AutoencoderKL.from_pretrained(
            self._pretrained_model_path,
            subfolder="vae"
        ).to(device)
        self._vae.requires_grad_(False)
        self._vae.eval()  # Ensure eval mode
        self.device = device
        self.deterministic = deterministic
        self.scaling_factor = 0.18215

    def forward(self, x):
        """
        Encode images to latent space

        Args:
            x: Input images, shape (B, 3, H, W), range [-1, 1]

        Returns:
            latents: Encoded latents, shape (B, 4, H//8, W//8)
        """
        with torch.no_grad():
            vae_output = self._vae.encode(x.to(self.device))

            # Use mode for deterministic behavior, sample for stochastic
            if self.deterministic:
                latents = vae_output.latent_dist.mode()
            else:
                latents = vae_output.latent_dist.sample()

            latents = latents * self.scaling_factor

        return latents

    def encode_batch(self, images, batch_size: int = 8):
        """
        Encode large batch of images in smaller chunks to save memory

        Args:
            images: Input images, shape (B, 3, H, W)
            batch_size: Size of sub-batches

        Returns:
            latents: Encoded latents
        """
        latents_list = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]
            latents = self.forward(batch)
            latents_list.append(latents)

        return torch.cat(latents_list, dim=0)
