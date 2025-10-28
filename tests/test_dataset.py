import pytest
import torch
from dataset.kttiDepthData import KttiDepthDataModule


@pytest.fixture(scope="module")
def data_module():
    """Fixture for data module"""
    return KttiDepthDataModule(batch_size=4, num_workers=2)


@pytest.fixture(scope="module")
def train_dataloader(data_module):
    """Fixture for train data loader"""
    return data_module.train_dataloader()


@pytest.fixture
def sample_batch(train_dataloader):
    """Fixture to get a sample batch"""
    return next(iter(train_dataloader))


class TestDataModule:
    """Tests for KttiDepthDataModule"""

    def test_data_module_initialization(self, data_module):
        """Test data module initializes correctly"""
        assert data_module is not None
        assert data_module.batch_size == 4
        assert data_module.num_workers == 2

    def test_train_dataloader_exists(self, train_dataloader):
        """Test train dataloader is created"""
        assert train_dataloader is not None

    def test_dataloader_is_iterable(self, train_dataloader):
        """Test dataloader can be iterated"""
        batch = next(iter(train_dataloader))
        assert batch is not None


class TestBatchFormat:
    """Tests for batch format and structure"""

    def test_batch_is_tuple(self, sample_batch):
        """Test batch returns tuple of (images, depth)"""
        assert isinstance(sample_batch, (tuple, list))
        assert len(sample_batch) == 2

    def test_batch_shapes(self, sample_batch):
        """Test batch has correct shapes"""
        images, depth = sample_batch

        assert images.ndim == 4  # (B, C, H, W)
        assert depth.ndim == 4
        assert images.size(0) == depth.size(0)  # Same batch size
        assert images.size(1) == 3  # RGB
        assert depth.size(1) == 3  # 3-channel depth

    def test_batch_size(self, sample_batch):
        """Test batch size matches configuration"""
        images, depth = sample_batch
        assert images.size(0) == 4  # Configured batch size

    def test_spatial_dimensions(self, sample_batch):
        """Test spatial dimensions are correct"""
        images, depth = sample_batch

        # Check height and width match
        assert images.size(2) == depth.size(2)  # Height
        assert images.size(3) == depth.size(3)  # Width

        # Check they're positive
        assert images.size(2) > 0
        assert images.size(3) > 0

    def test_batch_value_range(self, sample_batch):
        """Test batch values are normalized"""
        images, depth = sample_batch
        assert images.min() >= -1.5
        assert images.max() <= 1.5
        assert torch.isfinite(depth).all(), "Depth contains NaN or Inf"
        depth_median = depth.median()
        assert -5.0 < depth_median < 5.0, f"Depth median {depth_median} is extreme"

    def test_batch_dtype(self, sample_batch):
        """Test batch has correct dtype"""
        images, depth = sample_batch
        assert images.dtype == torch.float32
        assert depth.dtype == torch.float32

    def test_no_nan_values(self, sample_batch):
        """Test batch doesn't contain NaN values"""
        images, depth = sample_batch

        assert not torch.isnan(images).any()
        assert not torch.isnan(depth).any()

    def test_no_inf_values(self, sample_batch):
        """Test batch doesn't contain Inf values"""
        images, depth = sample_batch

        assert not torch.isinf(images).any()
        assert not torch.isinf(depth).any()


class TestMultipleBatches:
    """Tests for loading multiple batches"""

    def test_multiple_batches_load(self, data_module):
        """Test loading multiple batches"""
        dataloader = data_module.train_dataloader()
        batches = []

        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test 3 batches
                break
            images, depth = batch
            batches.append((images, depth))

            assert images.shape == batches[0][0].shape
            assert depth.shape == batches[0][1].shape

    def test_batches_are_different(self, data_module):
        """Test that different batches contain different data"""
        dataloader = data_module.train_dataloader()
        iterator = iter(dataloader)

        batch1 = next(iterator)
        batch2 = next(iterator)

        images1, _ = batch1
        images2, _ = batch2

        # Images should not be identical
        assert not torch.allclose(images1, images2)

    def test_consistent_shapes_across_batches(self, data_module):
        """Test all batches have consistent shapes"""
        dataloader = data_module.train_dataloader()

        first_batch = None
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Check first 5 batches
                break

            images, depth = batch

            if first_batch is None:
                first_batch = (images.shape, depth.shape)
            else:
                assert images.shape == first_batch[0]
                assert depth.shape == first_batch[1]


class TestDataTransforms:
    """Tests for data transformations"""

    def test_images_are_tensors(self, sample_batch):
        """Test images are torch tensors"""
        images, depth = sample_batch

        assert torch.is_tensor(images)
        assert torch.is_tensor(depth)

    def test_depth_has_three_channels(self, sample_batch):
        """Test depth is replicated to 3 channels"""
        _, depth = sample_batch
        assert depth.size(1) == 3


class TestDataLoaderProperties:
    """Tests for DataLoader configuration"""

    def test_dataloader_batch_size(self, train_dataloader):
        """Test dataloader batch size configuration"""
        batch = next(iter(train_dataloader))
        images, _ = batch
        assert images.size(0) <= 4  # Less than or equal to configured batch size

    def test_dataloader_provides_cpu_tensors_by_default(self, sample_batch):
        """Test dataloader returns CPU tensors by default"""
        images, depth = sample_batch

        # By default, DataLoader returns CPU tensors
        assert images.device.type == 'cpu'
        assert depth.device.type == 'cpu'
