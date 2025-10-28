from diffusers import DDIMScheduler, LCMScheduler
import torch
from torch import nn


class DDIMNoiseScheduler(nn.Module):
    """Wrapper for noise scheduler"""
    def __init__(self, pretrained_model_path: str = "stabilityai/stable-diffusion-2-base"):
        super().__init__()
        self.scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler"
        )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to original sample"""
        return self.scheduler.add_noise(original, noise, timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ):
        """Perform one denoising step"""
        return self.scheduler.step(model_output, timestep, sample)


class LCMNoiseScheduler(nn.Module):
    """Wrapper for noise scheduler"""
    def __init__(self, pretrained_model_path: str = "stabilityai/stable-diffusion-2-base"):
        super().__init__()
        self.scheduler = LCMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler"
        )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to original sample"""
        return self.scheduler.add_noise(original, noise, timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ):
        """Perform one denoising step"""
        return self.scheduler.step(model_output, timestep, sample)
