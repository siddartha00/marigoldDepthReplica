from diffusers import UNet2DConditionModel
from torch import nn


class LightweightUNet(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # 🧠 A much smaller UNet config (~45–55M params)
        self._unet = UNet2DConditionModel(
            sample_size=64,                     # typical latent size for SD-like models
            in_channels=8,                      # your modified input channels
            out_channels=4,                     # same as before
            down_block_types=(
                "DownBlock2D",                  # No attention here
                "DownBlock2D",                  # Keep simple
                "AttnDownBlock2D",              # Light attention at lowest level
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(128, 256, 512),  # 🪶 Much smaller than (320,640,1280,1280)
            layers_per_block=1,                  # Fewer layers per block
            cross_attention_dim=256,             # Light attention width
            attention_head_dim=4,                # Small attention heads
            norm_num_groups=16,                  # Smaller group norm
        ).to(device)

    def forward(self, x, timesteps, context):
        return self._unet(
            x.to(self.device),
            timesteps.to(self.device),
            encoder_hidden_states=context.to(self.device)
        ).sample
