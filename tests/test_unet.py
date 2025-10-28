import pytest
import torch
from marigold.modifiedUnet import ModifiedUNet
from marigold.latentEncoder import LatentEncoder
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def device():
    """Fixture to determine device availability"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def test_modified_unet(device):
    """Fixture for modified U-Net"""
    unet = ModifiedUNet(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device=device
    )
    return unet


@pytest.fixture(scope="module")
def test_latent_encoder(device):
    """Fixture for latent encoder (for integration tests)"""
    encoder = LatentEncoder(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device=device
    )
    return encoder


@pytest.fixture
def sample_batch():
    """Fixture to get a sample batch"""
    data_module = KttiDepthDataModule(batch_size=4, num_workers=2)
    dataloader = data_module.train_dataloader()
    return next(iter(dataloader))


class TestUNetInitialization:
    """Tests for ModifiedUNet initialization"""

    def test_unet_initialization(self, test_modified_unet, device):
        """Test U-Net initializes correctly"""
        assert test_modified_unet is not None
        assert hasattr(test_modified_unet, '_unet')
        assert test_modified_unet.device == device

    def test_unet_conv_in_channels(self, test_modified_unet):
        """Test first conv layer has 8 input channels"""
        conv_weight = test_modified_unet._unet.conv_in.weight
        assert conv_weight.shape[1] == 8

    def test_unet_conv_in_output_channels(self, test_modified_unet):
        """Test first conv has 320 output channels"""
        conv_weight = test_modified_unet._unet.conv_in.weight
        assert conv_weight.shape[0] == 320

    def test_unet_on_correct_device(self, test_modified_unet, device):
        """Test all U-Net parameters are on correct device"""
        for param in test_modified_unet._unet.parameters():
            assert param.device.type == device


class TestUNetForward:
    """Tests for U-Net forward pass"""

    def test_unet_forward_output_shape(self, test_modified_unet, device):
        """Test U-Net forward pass returns correct shape"""
        batch_size = 2
        x = torch.randn(batch_size, 8, 64, 64).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.zeros(batch_size, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)

        expected_shape = (batch_size, 4, 64, 64)
        assert output.shape == expected_shape

    def test_unet_forward_dtype(self, test_modified_unet, device):
        """Test U-Net output has correct dtype"""
        x = torch.randn(1, 8, 64, 64).to(device)
        timesteps = torch.tensor([500]).to(device)
        context = torch.zeros(1, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_unet_different_batch_sizes(self, test_modified_unet, device, batch_size):
        """Test U-Net handles various batch sizes"""
        x = torch.randn(batch_size, 8, 64, 64).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.zeros(batch_size, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)
        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("resolution", [32, 64, 96, 128])
    def test_unet_different_resolutions(self, test_modified_unet, device, resolution):
        """Test U-Net works with different spatial resolutions"""
        x = torch.randn(1, 8, resolution, resolution).to(device)
        timesteps = torch.tensor([500]).to(device)
        context = torch.zeros(1, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)
        assert output.shape == (1, 4, resolution, resolution)


class TestUNetWithRealData:
    """Tests for U-Net with real encoded data"""

    def test_unet_forward_with_real_data(
        self,
        test_modified_unet,
        test_latent_encoder,
        sample_batch,
        device
    ):
        """Test U-Net forward pass with real encoded data"""
        images, depth = sample_batch
        images = images.to(device)
        depth = depth.to(device)

        # Encode
        image_latents = test_latent_encoder(images)
        depth_latents = test_latent_encoder(depth)

        # Concatenate
        latent_input = torch.cat([depth_latents, image_latents], dim=1)

        # Create timesteps and context
        batch_size = latent_input.size(0)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.zeros(batch_size, 77, 1024).to(device)

        # Forward pass
        noise_pred = test_modified_unet(latent_input, timesteps, context)

        assert noise_pred.shape == (batch_size, 4, image_latents.shape[2], image_latents.shape[3])
        assert not torch.isnan(noise_pred).any()


class TestUNetGradients:
    """Tests for U-Net gradient computation"""

    def test_unet_gradient_flow(self, test_modified_unet, device):
        """Test gradients flow correctly through U-Net"""
        x = torch.randn(1, 8, 64, 64, device=device, requires_grad=True)
        timesteps = torch.tensor([500], device=device)
        context = torch.zeros(1, 77, 1024, device=device)

        # Zero gradients
        test_modified_unet._unet.zero_grad()

        output = test_modified_unet(x, timesteps, context)
        loss = output.sum()
        loss.backward()

        # Check that model parameters have gradients
        total_grad_norm = sum(
            p.grad.norm().item()
            for p in test_modified_unet._unet.parameters()
            if p.grad is not None
        )

        assert total_grad_norm > 0, "Gradients did not flow through model"

    def test_unet_parameters_trainable(self, test_modified_unet):
        """Test U-Net parameters require gradients"""
        trainable_params = [
            p for p in test_modified_unet._unet.parameters()
            if p.requires_grad
        ]
        assert len(trainable_params) > 0


class TestUNetEdgeCases:
    """Tests for U-Net edge cases and error handling"""

    def test_unet_wrong_input_channels_raises_error(self, test_modified_unet, device):
        """Test wrong number of input channels raises error"""
        x = torch.randn(1, 4, 64, 64).to(device)  # Wrong: 4 instead of 8
        timesteps = torch.tensor([500]).to(device)
        context = torch.zeros(1, 77, 1024).to(device)

        with pytest.raises(RuntimeError):
            test_modified_unet(x, timesteps, context)

    def test_unet_timestep_broadcasting_works(self, test_modified_unet, device):
        """Test that single timestep broadcasts to batch"""
        x = torch.randn(4, 8, 64, 64).to(device)
        timesteps = torch.tensor([500]).to(device)  # Single timestep
        context = torch.zeros(4, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)
        assert output.shape == (4, 4, 64, 64)

    def test_unet_zero_input(self, test_modified_unet, device):
        """Test U-Net handles zero input"""
        x = torch.zeros(1, 8, 64, 64).to(device)
        timesteps = torch.tensor([500]).to(device)
        context = torch.zeros(1, 77, 1024).to(device)

        output = test_modified_unet(x, timesteps, context)
        assert output.shape == (1, 4, 64, 64)
        assert not torch.isnan(output).any()


class TestUNetIntegration:
    """Integration tests for U-Net with encoder"""

    def test_full_forward_pipeline(
        self,
        test_latent_encoder,
        test_modified_unet,
        sample_batch,
        device
    ):
        """Test complete forward pipeline from data to noise prediction"""
        images, depth = sample_batch
        images = images.to(device)
        depth = depth.to(device)
        batch_size = images.size(0)

        # Step 1: Encode images and depth
        image_latents = test_latent_encoder(images)
        depth_latents = test_latent_encoder(depth)

        # Step 2: Concatenate latents
        latent_input = torch.cat([depth_latents, image_latents], dim=1)

        # Step 3: Prepare U-Net inputs
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.zeros(batch_size, 77, 1024).to(device)

        # Step 4: U-Net forward pass
        noise_pred = test_modified_unet(latent_input, timesteps, context)

        # Assertions
        assert noise_pred.shape == depth_latents.shape
        assert torch.is_tensor(noise_pred)
        assert not torch.isnan(noise_pred).any()
        assert not torch.isinf(noise_pred).any()

    def test_training_step_simulation(
        self,
        test_latent_encoder,
        test_modified_unet,
        sample_batch,
        device
    ):
        """Simulate a training step"""
        images, depth = sample_batch
        images = images.to(device)
        depth = depth.to(device)
        batch_size = images.size(0)

        # Encode
        image_latents = test_latent_encoder(images)
        depth_latents = test_latent_encoder(depth)

        # Add noise
        noise = torch.randn_like(depth_latents)
        noisy_depth = depth_latents + noise * 0.1

        # Concatenate
        latent_input = torch.cat([noisy_depth, image_latents], dim=1)

        # Forward pass
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        context = torch.zeros(batch_size, 77, 1024).to(device)

        noise_pred = test_modified_unet(latent_input, timesteps, context)

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        assert loss.item() >= 0
        assert not torch.isnan(loss)
