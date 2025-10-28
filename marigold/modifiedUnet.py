from diffusers import UNet2DConditionModel
from torch import nn
import torch


class ModifiedUNet(nn.Module):
    def __init__(self, pretrained_model_path: str = "stabilityai/stable-diffusion-2-base", device: str = "cuda"):
        super().__init__()
        self._pretrained_model_path = pretrained_model_path
        self.device = device
        orig_net = UNet2DConditionModel.from_pretrained(
            self._pretrained_model_path,
            subfolder="unet",
            layer_per_block=2
        )
        self._unet = UNet2DConditionModel.from_pretrained(
            self._pretrained_model_path,
            subfolder="unet",
            in_channels=8,
            out_channels=4,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            layer_per_block=2
        )
        orig_weights = orig_net.conv_in.weight.data
        new_weights = torch.cat([orig_weights, orig_weights], dim=1) * 0.5
        self._unet.conv_in.weight.data = new_weights.to(device)
        if orig_net.conv_in.bias is not None:
            self._unet.conv_in.bias.data = orig_net.conv_in.bias.data.clone().to(device)
        self._unet = self._unet.to(device)
        del orig_net

    def forward(self, x, timesteps, context):
        noise_pred = self._unet(
            x.to(self.device),
            timesteps.to(self.device),
            encoder_hidden_states=context.to(self.device)
        ).sample
        return noise_pred
