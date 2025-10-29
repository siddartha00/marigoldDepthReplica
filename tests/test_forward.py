# tests/test_forward.py

import pytest
import torch
from marigold.marigoldDepth import MarigoldDepth
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def marigold_model(device):
    model = MarigoldDepth(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device_type=device,
        learning_rate=1e-4,
        image_size=512,
        vae_downsample=8
    )
    model.to(device)
    return model


@pytest.fixture
def sample_batch():
    data_module = KttiDepthDataModule(batch_size=2, num_workers=2)
    dataloader = data_module.train_dataloader()
    return next(iter(dataloader))


class TestMarigoldInitialization:

    def test_model_initialization(self, marigold_model):
        assert marigold_model is not None
        assert hasattr(marigold_model, 'latent_encoder')
        assert hasattr(marigold_model, 'latent_decoder')
        assert hasattr(marigold_model, 'noise_pred')
        assert hasattr(marigold_model, 'scheduler')


class TestMarigoldTrainingStep:

    def test_training_step_runs(self, marigold_model, sample_batch, device):
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))
        loss = marigold_model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestMarigoldEdgeCases:

    def test_single_image_batch(self, marigold_model, device):
        image = torch.randn(1, 3, 512, 512).to(device)
        depth = torch.randn(1, 3, 512, 512).to(device)
        batch = (image, depth)
        loss = marigold_model.training_step(batch, batch_idx=0)
        assert not torch.isnan(loss)

    def test_different_image_sizes(self, marigold_model, device):
        """Dynamic latent size check"""
        for size in [256, 384, 512]:
            image = torch.randn(1, 3, size, size).to(device)
            expected_latent_size = size // marigold_model.hparams['vae_downsample']
            image_latents = marigold_model.latent_encoder(image)
            assert image_latents.shape[2] == expected_latent_size
            assert image_latents.shape[3] == expected_latent_size


class TestMarigoldPrediction:

    def test_predict_depth_runs(self, marigold_model, device):
        image = torch.randn(1, 3, 512, 512).to(device)
        depth = marigold_model.predict_depth(image, num_inference_steps=5, ensemble_size=1)
        assert depth.shape[0] == 1
        assert depth.shape[1] == 1

    def test_forward_calls_predict(self, marigold_model, device):
        image = torch.randn(1, 3, 512, 512).to(device)
        marigold_model.eval()
        with torch.no_grad():
            depth = marigold_model(image)
        assert depth.shape[1] == 1


class TestDiTSpecific:

    def test_dit_parameter_count(self, marigold_model):
        """DiT-Small has ~1M parameters"""
        total_params = sum(p.numel() for p in marigold_model.noise_pred.parameters())
        assert 900_000 < total_params < 1_500_000

    def test_dit_timestep_conditioning(self, marigold_model, device):
        image = torch.randn(1, 3, 512, 512).to(device)
        depth = torch.randn(1, 3, 512, 512).to(device)
        image_latents = marigold_model.latent_encoder(image)
        depth_latents = marigold_model.latent_encoder(depth)
        latent_input = torch.cat([depth_latents, image_latents], dim=1)
        t1 = torch.tensor([100], device=device)
        t2 = torch.tensor([500], device=device)
        with torch.no_grad():
            pred1 = marigold_model.noise_pred(latent_input, timestep=t1)
            pred2 = marigold_model.noise_pred(latent_input, timestep=t2)
        assert not torch.allclose(pred1, pred2, atol=1e-3)
