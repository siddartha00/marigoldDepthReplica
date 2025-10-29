# marigold/dit_noise_pred.py
import torch
from torch import nn, Tensor
from typing import Optional, Union
from diffusers.models import DiTTransformer2DModel


class DitNoisePred(nn.Module):
    """
    DiT-S based noise predictor to replace UNet2DConditionModel.

    Inputs:
      - latent_input: (B, 8, H, W)
      - timestep: scalar, tensor, or None
    Output:
      - Tensor (B, 4, H, W) predicted noise
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        sample_size: int = 64,
        num_layers: int = 6,
        embed_dim: int = 384,
        num_heads: int = 6,
        patch_size: int = 2,
        dropout: float = 0.0,
        device: str = "cuda"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size

        self.dit = DiTTransformer2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            num_layers=num_layers,
            attention_head_dim=embed_dim // num_heads,
            num_attention_heads=num_heads,
            patch_size=patch_size,
            dropout=dropout
        )

    # ✅ Fix: normalize timestep to proper tensor for DiT
    def _normalize_timestep(self, timestep, batch, device):
        if timestep is None:
            t = torch.zeros(batch, dtype=torch.long, device=device)
        elif isinstance(timestep, int):
            t = torch.full((batch,), timestep, dtype=torch.long, device=device)
        elif isinstance(timestep, Tensor):
            t = timestep.to(device)
            if t.dim() == 0:
                t = t.expand(batch)
            elif t.shape[0] == 1 and batch > 1:
                t = t.expand(batch)
            t = t.long()
        else:
            raise TypeError(f"Unsupported timestep type: {type(timestep)}")
        return t

    def forward(
        self,
        latent_input: Tensor,
        timestep: Optional[Union[int, Tensor]] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        return_dict: bool = False,
    ):
        b, c, h, w = latent_input.shape

        # ✅ Validate for clarity (keeps test expectations clean)
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        if h != self.sample_size or w != self.sample_size:
            raise ValueError(f"Expected {self.sample_size}x{self.sample_size}, got {h}x{w}")

        # Normalize timestep
        t = self._normalize_timestep(timestep, b, latent_input.device)

        batch_size = latent_input.shape[0]
        dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=latent_input.device)

        # Forward through DiT
        out = self.dit(hidden_states=latent_input, timestep=t, class_labels=dummy_labels)

        # ✅ Safely extract result
        if out is None:
            raise RuntimeError("DiT returned None")
        noise_pred = getattr(out, "sample", None)
        if noise_pred is None:
            raise RuntimeError("DiT output missing 'sample'")

        return {"sample": noise_pred} if return_dict else noise_pred
