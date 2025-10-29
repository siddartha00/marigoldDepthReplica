import pytest
import torch
from marigold.DiTNoisePred import DitNoisePred


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    # Small but DiT-S–like config for fast unit testing
    m = DitNoisePred(
        in_channels=8,
        out_channels=4,
        sample_size=64,
        num_layers=2,     # shallow for speed
        embed_dim=128,
        num_heads=4,
        patch_size=2,
        dropout=0.0
    ).to(device)
    return m


# ---------- Core Functionality Tests ----------

class TestDitNoisePredBasics:
    def test_forward_tensor_output(self, model, device):
        x = torch.randn(2, 8, 64, 64, device=device)
        t = torch.randint(0, 1000, (2,), device=device)
        y = model(x, timestep=t, return_dict=False)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 4, 64, 64)

    def test_forward_return_dict(self, model, device):
        x = torch.randn(1, 8, 64, 64, device=device)
        y = model(x, timestep=0, return_dict=True)
        assert isinstance(y, dict)
        assert "sample" in y
        assert y["sample"].shape == (1, 4, 64, 64)

    def test_scalar_timestep(self, model, device):
        x = torch.randn(1, 8, 64, 64, device=device)
        y = model(x, timestep=250)
        assert y.shape == (1, 4, 64, 64)

    def test_none_timestep_defaults_to_zero(self, model, device):
        # Model should accept None and default internally (handled by DiT)
        x = torch.randn(1, 8, 64, 64, device=device)
        y = model(x, timestep=None)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 4, 64, 64)

    def test_single_timestep_broadcasts(self, model, device):
        x = torch.randn(4, 8, 64, 64, device=device)
        # Single timestep tensor should broadcast across batch
        y = model(x, timestep=torch.tensor([10], device=device))
        assert y.shape == (4, 4, 64, 64)

    def test_different_batch_sizes(self, model, device):
        for bs in [1, 2, 4]:
            x = torch.randn(bs, 8, 64, 64, device=device)
            t = torch.randint(0, 1000, (bs,), device=device)
            y = model(x, timestep=t)
            assert y.shape == (bs, 4, 64, 64)


# ---------- Gradient & Stability Tests ----------

class TestGradientsAndStability:
    def test_grad_flow(self, model, device):
        model.train()
        x = torch.randn(2, 8, 64, 64, device=device, requires_grad=True)
        t = torch.randint(0, 1000, (2,), device=device)
        y = model(x, timestep=t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_no_nan_inf(self, model, device):
        model.eval()
        x = torch.randn(2, 8, 64, 64, device=device)
        t = torch.randint(0, 1000, (2,), device=device)
        with torch.no_grad():
            y = model(x, timestep=t)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed precision test")
    def test_mixed_precision_cuda(self, device):
        # Tests fp16 forward compatibility
        model = DitNoisePred(
            in_channels=8, out_channels=4, sample_size=64,
            num_layers=2, embed_dim=128, num_heads=4
        ).half().to(device)
        x = torch.randn(1, 8, 64, 64, dtype=torch.float16, device=device)
        y = model(x, timestep=0)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == torch.float16


# ---------- Optional Integration (Scheduler Smoke Test) ----------

@pytest.mark.slow
class TestIntegrationWithScheduler:
    def test_scheduler_step_smoke(self, device):
        try:
            from marigold.noise import DDIMNoiseScheduler
        except ImportError:
            pytest.skip("DDIMNoiseScheduler not available for test")

        model = DitNoisePred(
            in_channels=8, out_channels=4, sample_size=64,
            num_layers=2, embed_dim=128, num_heads=4
        ).to(device)

        sched = DDIMNoiseScheduler(pretrained_model_path="stabilityai/stable-diffusion-2-base")
        sched.scheduler.set_timesteps(num_inference_steps=50)


        b, h, w = 2, 64, 64
        img_lat = torch.randn(b, 4, h, w, device=device)
        depth_lat = torch.randn(b, 4, h, w, device=device)
        latent = torch.cat([depth_lat, img_lat], dim=1)
        t = torch.randint(0, sched.num_train_timesteps, (b,), device=device)

        with torch.no_grad():
            y = model(latent, timestep=t)
            step = sched.step(y, t[0], depth_lat)
            assert hasattr(step, "prev_sample")
            assert step.prev_sample.shape == depth_lat.shape
