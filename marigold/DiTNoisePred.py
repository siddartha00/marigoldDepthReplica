import torch
from torch import nn, Tensor
from typing import Optional, Union
from diffusers.models import DiTTransformer2DModel


class DitNoisePred(nn.Module):
    """
    DiT-S based noise predictor to replace UNet2DConditionModel.
    """

    def __init__(
        self,
        in_channels=8,
        out_channels=4,
        sample_size=64,
        num_layers=12,
        num_attention_heads=6,
        attention_head_dim=64,
        patch_size=2,
        dropout=0.0,
        device="cuda"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sample_size = sample_size

        self.dit = DiTTransformer2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=sample_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            patch_size=patch_size,
            dropout=dropout
        ).to(device)

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

        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        if h != self.sample_size or w != self.sample_size:
            raise ValueError(f"Expected {self.sample_size}x{self.sample_size}, got {h}x{w}")

        t = self._normalize_timestep(timestep, b, latent_input.device)
        dummy_labels = torch.zeros(b, dtype=torch.long, device=latent_input.device)

        out = self.dit(hidden_states=latent_input, timestep=t, class_labels=dummy_labels)
        noise_pred = getattr(out, "sample", None)
        if noise_pred is None:
            raise RuntimeError("DiT output missing 'sample'")

        return {"sample": noise_pred} if return_dict else noise_pred
