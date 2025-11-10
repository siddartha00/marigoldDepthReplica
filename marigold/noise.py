from diffusers import DDIMScheduler
import torch
from torch import nn


class MarigoldSchedulerWrapper(nn.Module):
    def __init__(self, pretrained_model_path: str, trailing_fraction: float = 1.0, zero_snr: bool = False):
        super().__init__()
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False
        )
        # Optionally modify betas to implement trailing timestamps
        if trailing_fraction > 0:
            T = self.scheduler.config.num_train_timesteps
            # only use last trailing_fraction of timesteps during training/inference
            cut = int(T * trailing_fraction)
            # For example: keep only timesteps [T-cut … T-1]
            self.scheduler.timesteps = self.scheduler.set_timesteps(T*trailing_fraction)
            self.scheduler.config.num_train_timesteps = cut

        if zero_snr:
            # scale beta schedule to drive SNR → 0 at final step
            # For simplicity you might amplify betas linearly near end
            betas = self.scheduler.betas.clone()
            T = betas.shape[0]
            # Example: linearly increase beta in final 20% of timesteps
            for i in range(int(0.8 * T), T):
                betas[i] = min(betas[i] * 2.0, 0.999)  # example scaling
            self.scheduler.betas = betas
            self.scheduler.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(original, noise, timesteps)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor):
        return self.scheduler.step(model_output, timestep, sample)
