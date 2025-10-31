# tests/test_noise_pred.py
import pytest
import torch
from marigold.DiTNoisePred import DitNoisePred


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    # Small DiT-S–like config for fast unit testing
    m = DitNoisePred(
        in_channels=8,
        out_channels=4,
        sample_size=64,
        num_layers=2,               # shallow for speed
        num_attention_heads=6,      # ✅ updated
        attention_head_dim=64,
        patch_size=2,
        dropout=0.0,
        device=str(device)
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
        x = torch.randn(1, 8, 64, 64, device=device)
        y = model(x, timestep=None)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (1, 4, 64, 64)

    def test_single_timestep_broadcasts(self, model, device):
        x = torch.randn(4, 8, 64, 64, device=device)
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
        m = DitNoisePred(
            in_channels=8,
            out_channels=4,
            sample_size=64,
            num_layers=2,
            num_attention_heads=6,  # ✅ updated
            attention_head_dim=64,
            device=str(device)
        ).to(device).half()

        x = torch.randn(1, 8, 64, 64, dtype=torch.float16, device=device)
        y = m(x, timestep=0)
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
            in_channels=8,
            out_channels=4,
            sample_size=64,
            num_layers=2,
            num_attention_heads=6,  # ✅ updated
            attention_head_dim=64,
            device=str(device)
        ).to(device)

        sched = DDIMNoiseScheduler(pretrained_model_path="stabilityai/stable-diffusion-2-base")

        if hasattr(sched, "scheduler") and hasattr(sched.scheduler, "set_timesteps"):
            sched.scheduler.set_timesteps(num_inference_steps=50)
        elif hasattr(sched, "set_timesteps"):
            sched.set_timesteps(num_inference_steps=50)
        else:
            pytest.skip("DDIMNoiseScheduler does not expose set_timesteps in this version")

        b, h, w = 2, 64, 64
        img_lat = torch.randn(b, 4, h, w, device=device)
        depth_lat = torch.randn(b, 4, h, w, device=device)
        latent = torch.cat([depth_lat, img_lat], dim=1)
        t = torch.randint(0, sched.num_train_timesteps, (b,), device=device)

        with torch.no_grad():
            y = model(latent, timestep=t)
            scalar_t = t[0].item() if isinstance(t, (torch.Tensor,)) else int(t)
            step_out = sched.step(y, scalar_t, depth_lat)
            assert hasattr(step_out, "prev_sample") or ("prev_sample" in getattr(step_out, "__dict__", {}))
            prev = getattr(step_out, "prev_sample", None)
            if prev is not None:
                assert prev.shape == depth_lat.shape
