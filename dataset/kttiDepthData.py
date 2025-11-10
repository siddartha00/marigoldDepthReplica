from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import csv
import os


class Csv2ImageDepthDataset:
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported.")
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range.")

        sample = self.data[idx]
        rgb_path = os.path.join(self.root_dir, sample['rgb_file'])
        depth_path = os.path.join(self.root_dir, sample['depth_file'])

        # Load PIL images
        image = self.load_image(rgb_path)
        depth = self.load_depth(depth_path)

        if self.transform:
            # Apply transforms (Resize, RandomHorizontalFlip, ToTensor)
            # After ToTensor, images are in [0, 1] range with shape (C, H, W)
            image = self.transform(image)
            depth = self.transform(depth)

            # Normalize image from [0, 1] to [-1, 1]
            image = image * 2.0 - 1.0

            # Normalize depth using percentile-based method
            # depth is a tensor now, convert to numpy for percentile calculation
            depth_np = depth.squeeze().numpy() if depth.dim() == 3 else depth.numpy()
            d2, d98 = np.percentile(depth_np, (2, 98))

            # Normalize depth to [-1, 1]
            depth = ((depth - d2) / (d98 - d2 + 1e-8) - 0.5) * 2.0

            # Ensure depth has 3 channels
            if depth.shape[0] == 1:
                depth = depth.repeat(3, 1, 1)
            elif depth.dim() == 2:
                depth = depth.unsqueeze(0).repeat(3, 1, 1)
        else:
            # No transform: normalize PIL images directly
            normalized_image = self.normalize_image(image)
            normalized_depth = self.normalize_depth(depth)

            image = torch.from_numpy(normalized_image).permute(2, 0, 1).float()
            depth = torch.from_numpy(normalized_depth).permute(2, 0, 1).float()

        return image, depth

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def load_depth(self, path):
        return Image.open(path)

    def normalize_depth(self, depth):
        """Normalize depth PIL image to [-1, 1] using percentile-based scaling"""
        depth_array = np.array(depth).astype('float32')
        d2, d98 = np.percentile(depth_array, (2, 98))
        normalized_depth_array = ((depth_array - d2) / (d98 - d2 + 1e-8) - 0.5) * 2
        # Replicate to 3 channels
        normalized_depth_array = np.stack([normalized_depth_array] * 3, axis=-1)
        return normalized_depth_array

    def normalize_image(self, image):
        """Normalize image PIL to [-1, 1]"""
        image_array = np.array(image).astype('float32') / 255.0 * 2.0 - 1.0
        return image_array


class KttiDepthDataModule(LightningDataModule):
    def __init__(self, batch_size=4, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = os.path.abspath(
            os.path.join(os.path.abspath(__file__), "../..")
        )
        self.data_dir = os.path.join(self.root_dir, 'data', 'virtual_kitti_2')
        self.csv_dir = os.path.join(self.data_dir, 'splits')

    def train_dataloader(self):
        train_csv_path = os.path.join(self.csv_dir, 'train.csv')
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
        ])
        train_dataset = Csv2ImageDepthDataset(
            csv_file=train_csv_path,
            root_dir=self.root_dir,
            transform=train_transform
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        val_csv_path = os.path.join(self.csv_dir, 'val.csv')
        # Define transform BEFORE using it
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),  # No random flips for validation
        ])
        val_dataset = Csv2ImageDepthDataset(
            csv_file=val_csv_path,
            root_dir=self.root_dir,
            transform=val_transform
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        test_csv_path = os.path.join(self.csv_dir, 'test.csv')
        test_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        test_dataset = Csv2ImageDepthDataset(
            csv_file=test_csv_path,
            root_dir=self.root_dir,
            transform=test_transform
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
