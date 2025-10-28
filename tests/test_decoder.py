import pytest
import torch
from marigold.latentDecoder import LatentDecoder
from marigold.latentEncoder import LatentEncoder
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def device():
    """Fixture to determine device availability"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def test_decoder(device):
    """Fixture for latent decoder"""
    decoder = LatentDecoder(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device=device
    )
    return decoder


@pytest.fixture(scope="module")
def test_encoder(device):
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


@pytest.fixture
def sample_latents(device):
    """Fixture for sample latent codes"""
    return torch.randn(2, 4, 64, 64).to(device)


class TestDecoderInitialization:
    """Tests for LatentDecoder initialization"""

    def test_decoder_initialization(self, test_decoder, device):
        """Test decoder initializes correctly"""
        assert test_decoder is not None
        assert hasattr(test_decoder, '_vae')
        assert test_decoder.device == device

    def test_decoder_has_vae(self, test_decoder):
        """Test decoder has VAE model"""
        assert hasattr(test_decoder, '_vae')
        assert test_decoder._vae is not None

    def test_decoder_on_correct_device(self, test_decoder, device):
        """Test decoder is on correct device"""
        for param in test_decoder._vae.parameters():
            assert param.device.type == device

    def test_decoder_has_scaling_factor(self, test_decoder):
        """Test decoder has scaling factor attribute"""
        assert hasattr(test_decoder, 'scaling_factor')
        assert test_decoder.scaling_factor == 0.18215


class TestDecoderForward:
    """Tests for decoder forward pass"""

    def test_decoder_forward_basic(self, test_decoder, sample_latents):
        """Test decoder processes latents correctly"""
        images = test_decoder(sample_latents)

        assert images.ndim == 4
        assert images.size(0) == sample_latents.size(0)  # Batch size preserved
        assert images.size(1) == 3  # RGB channels
        assert torch.is_tensor(images)
        assert not torch.isnan(images).any()

    def test_decoder_output_shape(self, test_decoder, device):
        """Test decoder output has correct shape"""
        latents = torch.randn(4, 4, 32, 32).to(device)
        images = test_decoder(latents)

        # Images should be upsampled by factor of 8
        expected_height = latents.size(2) * 8
        expected_width = latents.size(3) * 8

        assert images.shape == (4, 3, expected_height, expected_width)

    def test_decoder_output_dtype(self, test_decoder, sample_latents):
        """Test decoder output has correct dtype"""
        images = test_decoder(sample_latents)
        assert images.dtype == torch.float32

    def test_decoder_output_device(self, test_decoder, sample_latents, device):
        """Test decoder output is on correct device"""
        images = test_decoder(sample_latents)
        assert images.device.type == device

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_decoder_different_batch_sizes(self, test_decoder, device, batch_size):
        """Test decoder handles various batch sizes"""
        latents = torch.randn(batch_size, 4, 64, 64).to(device)
        images = test_decoder(latents)
        assert images.shape[0] == batch_size

    @pytest.mark.parametrize("latent_size", [16, 32, 64, 96])
    def test_decoder_different_latent_sizes(self, test_decoder, device, latent_size):
        """Test decoder works with different latent spatial sizes"""
        latents = torch.randn(2, 4, latent_size, latent_size).to(device)
        images = test_decoder(latents)

        expected_size = latent_size * 8
        assert images.shape[2] == expected_size
        assert images.shape[3] == expected_size


class TestDecoderDepthConversion:
    """Tests for decode_to_depth method"""

    def test_decode_to_depth_basic(self, test_decoder, sample_latents):
        """Test decoding to depth map"""
        depth = test_decoder.decode_to_depth(sample_latents)

        assert depth.ndim == 4
        assert depth.size(0) == sample_latents.size(0)
        assert depth.size(1) == 1  # Single channel
        assert torch.is_tensor(depth)

    def test_decode_to_depth_shape(self, test_decoder, device):
        """Test decode_to_depth output shape"""
        latents = torch.randn(2, 4, 64, 64).to(device)
        depth = test_decoder.decode_to_depth(latents)

        assert depth.shape == (2, 1, 512, 512)  # 64 * 8 = 512

    def test_decode_to_depth_no_nan(self, test_decoder, sample_latents):
        """Test decode_to_depth doesn't produce NaN"""
        depth = test_decoder.decode_to_depth(sample_latents)
        assert not torch.isnan(depth).any()


class TestDecoderFrozen:
    """Tests that decoder is frozen"""

    def test_decoder_parameters_frozen(self, test_decoder):
        """Test that decoder parameters are frozen"""
        for param in test_decoder._vae.parameters():
            assert not param.requires_grad

    def test_decoder_in_eval_mode(self, test_decoder):
        """Test decoder is in evaluation mode"""
        assert not test_decoder._vae.training


class TestDecoderDeterminism:
    """Tests for decoder determinism"""

    def test_decoder_deterministic_output(self, test_decoder, sample_latents):
        """Test decoder produces same output for same input"""
        with torch.no_grad():
            images1 = test_decoder(sample_latents)
            images2 = test_decoder(sample_latents)

        # Decoder should be deterministic (no sampling)
        assert torch.allclose(images1, images2, rtol=1e-6)

    def test_decode_to_depth_deterministic(self, test_decoder, sample_latents):
        """Test decode_to_depth is deterministic"""
        with torch.no_grad():
            depth1 = test_decoder.decode_to_depth(sample_latents)
            depth2 = test_decoder.decode_to_depth(sample_latents)

        assert torch.allclose(depth1, depth2, rtol=1e-6)


class TestEncoderDecoderRoundTrip:
    """Tests for encoder-decoder round trip"""

    def test_encode_decode_roundtrip(self, test_encoder, test_decoder, device):
        """Test encoding then decoding produces similar images"""
        # Create test image
        original_images = torch.randn(2, 3, 512, 512).to(device)

        # Encode
        latents = test_encoder(original_images)

        # Decode
        reconstructed_images = test_decoder(latents)

        # Check shapes match
        assert reconstructed_images.shape == original_images.shape

        # Reconstruction won't be perfect, but should be similar
        # Check that they're correlated
        mse = torch.nn.functional.mse_loss(
            reconstructed_images,
            original_images
        ).item()

        # MSE should be reasonable (not too high)
        assert mse < 10.0, f"MSE too high: {mse}"

    def test_encode_decode_with_real_data(
        self,
        test_encoder,
        test_decoder,
        sample_batch,
        device
    ):
        """Test round trip with real data"""
        images, _ = sample_batch
        images = images.to(device)

        # Encode
        latents = test_encoder(images)

        # Decode
        reconstructed = test_decoder(latents)

        assert reconstructed.shape == images.shape
        assert torch.isfinite(reconstructed).all()


class TestDecoderEdgeCases:
    """Tests for decoder edge cases"""

    def test_decoder_single_sample(self, test_decoder, device):
        """Test decoder with single sample"""
        latents = torch.randn(1, 4, 64, 64).to(device)
        images = test_decoder(latents)

        assert images.shape[0] == 1
        assert images.shape[1] == 3

    def test_decoder_zero_latents(self, test_decoder, device):
        """Test decoder with zero latents"""
        latents = torch.zeros(2, 4, 64, 64).to(device)
        images = test_decoder(latents)

        assert images.shape == (2, 3, 512, 512)
        assert not torch.isnan(images).any()

    def test_decoder_handles_negative_latents(self, test_decoder, device):
        """Test decoder handles negative latent values"""
        latents = torch.randn(2, 4, 64, 64).to(device) - 2.0
        images = test_decoder(latents)

        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()

    def test_decoder_large_latent_values(self, test_decoder, device):
        """Test decoder with large latent values"""
        latents = torch.randn(2, 4, 64, 64).to(device) * 5.0
        images = test_decoder(latents)

        assert not torch.isnan(images).any()
        assert not torch.isinf(images).any()


class TestDecoderOutputProperties:
    """Tests for decoder output properties"""

    def test_decoder_output_value_range(self, test_decoder, sample_latents):
        """Test decoder output is in reasonable range"""
        images = test_decoder(sample_latents)

        # VAE output should be roughly in [-1, 1] range, but can exceed
        assert images.min() >= -5.0
        assert images.max() <= 5.0

    def test_decode_to_depth_value_range(self, test_decoder, sample_latents):
        """Test decode_to_depth output range"""
        depth = test_decoder.decode_to_depth(sample_latents)

        # Averaged channels should be in similar range
        assert depth.min() >= -5.0
        assert depth.max() <= 5.0

    def test_decoder_no_gradient_computation(self, test_decoder, device):
        """Test that decoder doesn't compute gradients"""
        latents = torch.randn(2, 4, 64, 64, requires_grad=True).to(device)

        images = test_decoder(latents)

        # Decoder uses torch.no_grad(), so output shouldn't require grad
        assert not images.requires_grad


class TestDecoderPerformance:
    """Tests for decoder performance characteristics"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_decoder_no_memory_leak(self, test_decoder):
        """Test decoder doesn't leak memory"""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        for _ in range(10):
            latents = torch.randn(2, 4, 64, 64).cuda()
            images = test_decoder(latents)
            del latents, images

        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        memory_increase = final_memory - initial_memory
        assert memory_increase < 50 * 1024 * 1024, \
            f"Memory leak detected: {memory_increase / 1024 / 1024:.2f} MB"

    def test_decoder_consistent_performance(self, test_decoder, device):
        """Test decoder performs consistently across multiple calls"""
        latents = torch.randn(4, 4, 64, 64).to(device)

        outputs = []
        for _ in range(5):
            output = test_decoder(latents)
            outputs.append(output)

        # All outputs should be identical
        for i in range(len(outputs) - 1):
            assert torch.allclose(outputs[i], outputs[i+1], rtol=1e-6)


class TestDecoderIntegration:
    """Integration tests for decoder with other components"""

    def test_decoder_with_unet_output(self, test_decoder, device):
        """Test decoder works with U-Net style output"""

        # Create mock U-Net output (noise prediction)
        noise_pred = torch.randn(2, 4, 64, 64).to(device)

        # Decoder should handle it
        images = test_decoder(noise_pred)
        assert images.shape == (2, 3, 512, 512)

    def test_full_pipeline_with_real_data(
        self,
        test_encoder,
        test_decoder,
        sample_batch,
        device
    ):
        """Test full encode-process-decode pipeline"""
        images, depths = sample_batch
        images = images.to(device)
        depths = depths.to(device)

        # Encode both
        depth_latents = test_encoder(depths)

        # Decode depth latents
        decoded_depth = test_decoder.decode_to_depth(depth_latents)

        # Should produce valid single-channel output
        assert decoded_depth.shape[0] == depths.shape[0]
        assert decoded_depth.shape[1] == 1
        assert torch.isfinite(decoded_depth).all()
