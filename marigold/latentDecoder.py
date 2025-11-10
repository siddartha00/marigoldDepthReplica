from diffusers import AutoencoderKL
import torch
from torch import nn


class LatentDecoder(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str = "stabilityai/stable-diffusion-2-base",
        device: str = "cuda"
    ):
        """
        VAE decoder for Marigold depth estimation

        Args:
            pretrained_model_path: Path to pretrained Stable Diffusion model
            device: Device to run model on ('cuda' or 'cpu')
        """
        super().__init__()
        self._pretrained_model_path = pretrained_model_path
        self._vae = AutoencoderKL.from_pretrained(
            self._pretrained_model_path,
            subfolder="vae"
        ).to(device)
        self._vae.requires_grad_(False)
        self._vae.eval()
        self.device = device
        self.scaling_factor = 0.18215

    def forward(self, latents):
        """
        Decode latents to image space

        Args:
            latents: Latent codes, shape (B, 4, h, w)

        Returns:
            images: Decoded images, shape (B, 3, H, W), range [-1, 1]
        """
        with torch.no_grad():
            images = self._vae.decode(latents.to(self.device) / self.scaling_factor).sample

        return images

    def decode_to_depth(self, latents):
        """
        Decode latents and convert to single-channel depth map
        Args:
            latents: Latent codes, shape (B, 4, h, w)
        Returns:
            depth: Depth map, shape (B, 1, H, W)
        """
        images = self.forward(latents)
        # Average RGB channels to get single-channel depth
        depth = images.mean(dim=1, keepdim=True)
        return depth
