import pytest
import torch
from marigold.latentEncoder import LatentEncoder
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def device():
    """Fixture to determine device availability"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def test_latent_encoder(device):
    """Fixture for latent encoder"""
    encoder = LatentEncoder(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device=device
    )
    return encoder


@pytest.fixture(scope="module")
def data_module():
    """Fixture for data module"""
    return KttiDepthDataModule(batch_size=4, num_workers=2)


@pytest.fixture
def sample_batch(data_module):
    """Fixture to get a sample batch"""
    dataloader = data_module.train_dataloader()
    return next(iter(dataloader))


class TestEncoderInitialization:
    """Tests for LatentEncoder initialization"""

    def test_encoder_initialization(self, test_latent_encoder, device):
        """Test encoder initializes correctly"""
        assert test_latent_encoder is not None
        assert hasattr(test_latent_encoder, '_vae')
        assert test_latent_encoder.device == device

    def test_encoder_has_vae(self, test_latent_encoder):
        """Test encoder has VAE model"""
        assert hasattr(test_latent_encoder, '_vae')
        assert test_latent_encoder._vae is not None

    def test_encoder_on_correct_device(self, test_latent_encoder, device):
        """Test encoder is on correct device"""
        for param in test_latent_encoder._vae.parameters():
            assert param.device.type == device


class TestEncoderForward:
    """Tests for encoder forward pass"""

    def test_encoder_forward_image(self, test_latent_encoder, sample_batch):
        """Test encoder processes images correctly"""
        images, _ = sample_batch
        images = images.to(test_latent_encoder.device)
        latents = test_latent_encoder(images)

        assert latents.ndim == 4
        assert latents.size(0) == images.size(0)  # Batch size preserved
        assert latents.size(1) == 4  # Latent channels
        assert torch.is_tensor(latents)
        assert not torch.isnan(latents).any()

    def test_encoder_forward_depth(self, test_latent_encoder, sample_batch):
        """Test encoder processes depth maps correctly"""
        _, depth = sample_batch
        depth = depth.to(test_latent_encoder.device)
        latents = test_latent_encoder(depth)

        assert latents.ndim == 4
        assert latents.size(0) == depth.size(0)
        assert latents.size(1) == 4
        assert torch.is_tensor(latents)

    def test_encoder_output_dtype(self, test_latent_encoder, sample_batch):
        """Test encoder output has correct dtype"""
        images, _ = sample_batch
        images = images.to(test_latent_encoder.device)
        latents = test_latent_encoder(images)
        assert latents.dtype == torch.float32

    def test_encoder_output_device(self, test_latent_encoder, sample_batch, device):
        """Test encoder output is on correct device"""
        images, _ = sample_batch
        images = images.to(device)
        latents = test_latent_encoder(images)
        assert latents.device.type == device


class TestEncoderLatentProperties:
    """Tests for latent properties"""

    def test_latent_dimensions(self, test_latent_encoder, sample_batch):
        """Test latent dimensions are correct"""
        images, _ = sample_batch
        images = images.to(test_latent_encoder.device)
        latents = test_latent_encoder(images)

        # Latent space should be downsampled by factor of 8
        assert latents.size(2) == images.size(2) // 8
        assert latents.size(3) == images.size(3) // 8

    def test_encoder_latent_concatenation(self, test_latent_encoder, sample_batch):
        """Test concatenating image and depth latents"""
        images, depth = sample_batch
        images = images.to(test_latent_encoder.device)
        depth = depth.to(test_latent_encoder.device)

        image_latents = test_latent_encoder(images)
        depth_latents = test_latent_encoder(depth)

        latent_input = torch.cat([depth_latents, image_latents], dim=1)

        assert latent_input.ndim == 4
        assert latent_input.size(1) == 8  # 4 depth + 4 image
        assert torch.is_tensor(latent_input)

    def test_latent_dimensions_match(self, test_latent_encoder, sample_batch):
        """Test that image and depth latents have matching dimensions"""
        images, depth = sample_batch
        images = images.to(test_latent_encoder.device)
        depth = depth.to(test_latent_encoder.device)

        image_latents = test_latent_encoder(images)
        depth_latents = test_latent_encoder(depth)

        assert image_latents.shape == depth_latents.shape
        assert image_latents.size(1) == 4
        assert depth_latents.size(1) == 4


class TestEncoderFrozen:
    """Tests that encoder is frozen"""

    def test_encoder_parameters_frozen(self, test_latent_encoder):
        """Test that encoder parameters are frozen"""
        for param in test_latent_encoder._vae.parameters():
            assert not param.requires_grad

    def test_encoder_in_eval_mode(self, test_latent_encoder):
        """Test encoder is in evaluation mode"""
        assert not test_latent_encoder._vae.training


class TestEncoderEdgeCases:
    """Tests for encoder edge cases"""

    def test_encoder_single_image(self, test_latent_encoder, device):
        """Test encoder with single image"""
        image = torch.randn(1, 3, 512, 512).to(device)
        latents = test_latent_encoder(image)

        assert latents.shape[0] == 1
        assert latents.shape[1] == 4

    def test_encoder_different_batch_sizes(self, test_latent_encoder, device):
        """Test encoder with different batch sizes"""
        for batch_size in [1, 2, 4, 8]:
            images = torch.randn(batch_size, 3, 512, 512).to(device)
            latents = test_latent_encoder(images)
            assert latents.shape[0] == batch_size
