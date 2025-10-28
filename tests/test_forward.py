# tests/test_marigold.py

import pytest
import torch
from marigold.marigoldDepth import MarigoldDepth
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def device():
    """Fixture to determine device availability"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def marigold_model(device):
    """Fixture for Marigold model"""
    model = MarigoldDepth(
        pretrained_model_path="stabilityai/stable-diffusion-2-base",
        device_type=device,
        learning_rate=1e-4
    )
    return model


@pytest.fixture
def sample_batch():
    """Fixture to get a sample batch"""
    data_module = KttiDepthDataModule(batch_size=2, num_workers=2)
    dataloader = data_module.train_dataloader()
    return next(iter(dataloader))


class TestMarigoldInitialization:
    """Tests for MarigoldDepth initialization"""

    def test_model_initialization(self, marigold_model):
        """Test model initializes correctly"""
        assert marigold_model is not None
        assert hasattr(marigold_model, 'latent_encoder')
        assert hasattr(marigold_model, 'latent_decoder')
        assert hasattr(marigold_model, 'modified_unet')
        assert hasattr(marigold_model, 'scheduler')

    def test_components_initialized(self, marigold_model):
        """Test all components are initialized"""
        assert marigold_model.latent_encoder is not None
        assert marigold_model.latent_decoder is not None
        assert marigold_model.modified_unet is not None
        assert marigold_model.scheduler is not None

    def test_hyperparameters_saved(self, marigold_model):
        """Test hyperparameters are saved"""
        assert hasattr(marigold_model, 'hparams')
        assert 'learning_rate' in marigold_model.hparams


class TestMarigoldTrainingStep:
    """Tests for training step"""

    def test_training_step_runs(self, marigold_model, sample_batch, device):
        """Test training step executes without error"""
        images, depths = sample_batch
        images = images.to(device)
        depths = depths.to(device)

        batch = (images, depths)
        loss = marigold_model.training_step(batch, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_training_step_loss_positive(self, marigold_model, sample_batch, device):
        """Test training loss is positive"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        loss = marigold_model.training_step(batch, batch_idx=0)
        assert loss.item() >= 0

    def test_training_step_backward(self, marigold_model, sample_batch, device):
        """Test backward pass works"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        marigold_model.modified_unet._unet.zero_grad()
        loss = marigold_model.training_step(batch, batch_idx=0)
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in marigold_model.modified_unet._unet.parameters():
            if param.grad is not None and not torch.all(param.grad == 0):
                has_grad = True
                break
        assert has_grad


class TestMarigoldValidationStep:
    """Tests for validation step"""

    def test_validation_step_runs(self, marigold_model, sample_batch, device):
        """Test validation step executes"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        loss = marigold_model.validation_step(batch, batch_idx=0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_validation_step_no_gradients(self, marigold_model, sample_batch, device):
        """Test validation doesn't update gradients"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        marigold_model.eval()
        with torch.no_grad():
            loss = marigold_model.validation_step(batch, batch_idx=0)

        assert not loss.requires_grad


class TestMarigoldPrediction:
    """Tests for prediction/inference"""

    def test_predict_depth_runs(self, marigold_model, device):
        """Test predict_depth executes"""
        image = torch.randn(1, 3, 512, 512).to(device)

        depth = marigold_model.predict_depth(
            image,
            num_inference_steps=5,  # Use few steps for speed
            ensemble_size=1
        )

        assert depth is not None
        assert depth.shape[0] == 1
        assert depth.shape[1] == 1  # Single channel depth

    def test_predict_depth_shape(self, marigold_model, device):
        """Test prediction output shape"""
        batch_size = 2
        image = torch.randn(batch_size, 3, 512, 512).to(device)

        depth = marigold_model.predict_depth(
            image,
            num_inference_steps=5,
            ensemble_size=1
        )

        assert depth.shape == (batch_size, 1, 512, 512)

    def test_predict_depth_ensemble(self, marigold_model, device):
        """Test ensemble prediction"""
        image = torch.randn(1, 3, 512, 512).to(device)

        depth = marigold_model.predict_depth(
            image,
            num_inference_steps=5,
            ensemble_size=3
        )

        assert depth.shape == (1, 1, 512, 512)
        assert not torch.isnan(depth).any()

    def test_forward_calls_predict(self, marigold_model, device):
        """Test forward method calls predict_depth"""
        image = torch.randn(1, 3, 512, 512).to(device)

        marigold_model.eval()
        with torch.no_grad():
            depth = marigold_model(image)

        assert depth is not None
        assert depth.shape[1] == 1


class TestMarigoldPredictStep:
    """Tests for predict_step method"""

    def test_predict_step_with_tuple(self, marigold_model, sample_batch, device):
        """Test predict_step with tuple input"""
        images, _ = sample_batch
        batch = (images.to(device),)

        predictions = marigold_model.predict_step(batch, batch_idx=0)

        assert predictions is not None
        assert predictions.shape[0] == images.shape[0]

    def test_predict_step_with_tensor(self, marigold_model, device):
        """Test predict_step with tensor input"""
        images = torch.randn(2, 3, 512, 512).to(device)

        predictions = marigold_model.predict_step(images, batch_idx=0)

        assert predictions is not None


class TestMarigoldOptimizer:
    """Tests for optimizer configuration"""

    def test_configure_optimizers(self, marigold_model):
        """Test optimizer is configured"""
        optimizer = marigold_model.configure_optimizers()

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_optimizer_learning_rate(self, marigold_model):
        """Test optimizer has correct learning rate"""
        optimizer = marigold_model.configure_optimizers()

        assert optimizer.param_groups[0]['lr'] == marigold_model.learning_rate

    def test_only_unet_optimized(self, marigold_model):
        """Test only U-Net parameters are optimized"""
        optimizer = marigold_model.configure_optimizers()

        # Get optimizer parameters
        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}

        # Check encoder params are NOT in optimizer
        for param in marigold_model.latent_encoder.parameters():
            assert id(param) not in opt_param_ids

        # Check decoder params are NOT in optimizer
        for param in marigold_model.latent_decoder.parameters():
            assert id(param) not in opt_param_ids


class TestMarigoldEdgeCases:
    """Tests for edge cases"""

    def test_single_image_batch(self, marigold_model, device):
        """Test with batch size 1"""
        image = torch.randn(1, 3, 512, 512).to(device)
        depth = torch.randn(1, 3, 512, 512).to(device)
        batch = (image, depth)

        loss = marigold_model.training_step(batch, batch_idx=0)
        assert not torch.isnan(loss)

    def test_different_image_sizes(self, marigold_model, device):
        """Test with different spatial dimensions"""
        for size in [256, 384, 512]:
            image = torch.randn(1, 3, size, size).to(device)
            depth = marigold_model.predict_depth(
                image,
                num_inference_steps=2,
                ensemble_size=1
            )
            assert depth.shape[2] == size
            assert depth.shape[3] == size


class TestMarigoldIntegration:
    """Integration tests"""

    def test_full_training_loop_simulation(self, marigold_model, sample_batch, device):
        """Simulate a full training iteration"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        # Training step
        marigold_model.train()
        loss = marigold_model.training_step(batch, batch_idx=0)

        # Backward
        marigold_model.modified_unet._unet.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer = marigold_model.configure_optimizers()
        optimizer.step()

        assert True  # If we get here, full loop works

    def test_train_then_predict(self, marigold_model, sample_batch, device):
        """Test training then switching to prediction"""
        images, depths = sample_batch
        batch = (images.to(device), depths.to(device))

        # Train
        marigold_model.train()
        train_loss = marigold_model.training_step(batch, batch_idx=0)

        # Predict
        marigold_model.eval()
        with torch.no_grad():
            prediction = marigold_model.predict_depth(
                images.to(device),
                num_inference_steps=2,
                ensemble_size=1
            )

        assert train_loss is not None
        assert prediction is not None
