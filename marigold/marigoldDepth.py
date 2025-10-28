from .latentDecoder import LatentDecoder
from .latentEncoder import LatentEncoder
from .modifiedUnet import ModifiedUNet
from .noise import DDIMNoiseScheduler
import pytorch_lightning as pl
import torch


class MarigoldDepth(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_path: str = "stabilityai/stable-diffusion-2-base",
        device_type: str = "cuda",
        learning_rate: float = 1e-4,
        number_of_training_steps: int = 100000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.device_type = device_type
        self.learning_rate = learning_rate
        self.number_of_training_steps = number_of_training_steps

        self.latent_encoder = LatentEncoder(
            pretrained_model_path=pretrained_model_path,
            device=self.device_type
        )
        self.latent_decoder = LatentDecoder(
            pretrained_model_path=pretrained_model_path,
            device=self.device_type
        )
        self.modified_unet = ModifiedUNet(
            pretrained_model_path=pretrained_model_path,
            device=self.device_type
        )
        self.scheduler = DDIMNoiseScheduler(
            pretrained_model_path=pretrained_model_path
        )

    def training_step(self, batch, batch_idx, *args, **kwargs):
        image, depth = batch
        batch_size = image.size(0)

        # Encode
        image_latents = self.latent_encoder(image)
        depth_latents = self.latent_encoder(depth)

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        # Add noise
        noise = torch.randn_like(depth_latents)
        noisy_depth_latents = self.scheduler.add_noise(
            depth_latents,
            noise,
            timesteps
        )

        # Concatenate [depth, image]
        latent_input = torch.cat([noisy_depth_latents, image_latents], dim=1)

        # Empty context (SD v2 uses 1024)
        empty_context = torch.zeros(batch_size, 77, 1024, device=self.device)

        # Predict noise
        noise_pred = self.modified_unet(
            latent_input,
            timesteps,
            empty_context
        )

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        image, depth = batch
        batch_size = image.size(0)

        image_latents = self.latent_encoder(image)
        depth_latents = self.latent_encoder(depth)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        noise = torch.randn_like(depth_latents)
        noisy_depth_latents = self.scheduler.add_noise(
            depth_latents,
            noise,
            timesteps
        )

        latent_input = torch.cat([noisy_depth_latents, image_latents], dim=1)
        empty_context = torch.zeros(batch_size, 77, 1024, device=self.device)

        noise_pred = self.modified_unet(
            latent_input,
            timesteps,
            empty_context
        )

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def forward(self, x):
        return self.predict_depth(x)

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        depth_predictions = self.predict_depth(images)
        return depth_predictions

    def predict_depth(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 50,
        ensemble_size: int = 10
    ):
        """Predict depth using iterative denoising"""
        self.modified_unet._unet.eval()

        with torch.no_grad():
            batch_size = image.size(0)
            # Get device from input image
            device = image.device

            image_latents = self.latent_encoder(image)
            predictions = []

            for _ in range(ensemble_size):
                # Initialize random noise on correct device
                depth_latent = torch.randn(
                    batch_size, 4,
                    image_latents.shape[2],
                    image_latents.shape[3],
                    device=device,
                    dtype=image.dtype
                )

                # Set timesteps
                self.scheduler.scheduler.set_timesteps(num_inference_steps)

                # Denoising loop
                for t in self.scheduler.scheduler.timesteps:
                    # Concatenate [depth, image]
                    latent_input = torch.cat([depth_latent, image_latents], dim=1)
                    empty_context = torch.zeros(
                        batch_size, 77, 1024,
                        device=device
                    )

                    # Predict noise
                    noise_pred = self.modified_unet(
                        latent_input,
                        t.unsqueeze(0).repeat(batch_size).to(device),
                        empty_context
                    )

                    # Denoise step
                    depth_latent = self.scheduler.step(
                        noise_pred,
                        t,
                        depth_latent
                    ).prev_sample

                # Decode
                depth_image = self.latent_decoder(depth_latent)

                # Average channels to get single-channel depth
                depth_map = depth_image.mean(dim=1, keepdim=True)
                predictions.append(depth_map)

            # Ensemble
            if ensemble_size > 1:
                final_depth = torch.median(torch.stack(predictions), dim=0).values
            else:
                final_depth = predictions[0]

            return final_depth

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.modified_unet._unet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-3  # optional tweak
        )

        reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=5e-4,
            cooldown=1,
            min_lr=1e-7,
        )

        scheduler = {
            "scheduler": reduce_on_plateau,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
            "name": "reduce_on_plateau",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_end(self):
        if self.trainer is not None and len(self.trainer.optimizers) > 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", current_lr, prog_bar=True, on_epoch=True)
