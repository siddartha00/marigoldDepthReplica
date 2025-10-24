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
        image = self.load_image(rgb_path)
        normalized_image = self.normalize_image(image)
        normalized_image = torch.from_numpy(normalized_image).permute(2, 0, 1)
        depth = self.load_depth(depth_path)
        normalized_depth = self.normalize_depth(depth)
        normalized_depth = torch.from_numpy(normalized_depth).permute(2, 0, 1)
        if self.transform:
            image = self.transform(normalized_image)
            depth = self.transform(normalized_depth)
        return image, depth

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def load_depth(self, path):
        return Image.open(path)

    def normalize_depth(self, depth):
        depth_array = np.array(depth).astype('float16')
        d2, d98 = np.percentile(depth_array, (2, 98))
        normalized_depth_array = ((depth_array - d2) / (d98 - d2) - 0.5) * 2
        normalized_depth_array = np.stack([normalized_depth_array]*3, axis=-1)
        return normalized_depth_array

    def normalize_image(self, image):
        image_array = np.array(image).astype('float16') / 255.0*2.0 - 1.0
        normalized_image_array = image_array.astype('float16')
        return normalized_image_array


class KttiDepthDataModule(LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = os.path.abspath(
            os.path.join(os.path.abspath(__file__), "../..")
        )
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.csv_dir = os.path.join(self.data_dir, 'splits')

    def train_dataloader(self):
        train_csv_path = os.path.join(self.csv_dir, 'train.csv')
        self.train_dataset = Csv2ImageDepthDataset(
            csv_file=train_csv_path,
            root_dir=self.root_dir
        )
        self.train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.random.HorizontalFlip(),
        ])
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          transform=self.train_transform)

    def val_dataloader(self):
        val_csv_path = os.path.join(self.csv_dir, 'val.csv')
        self.val_dataset = Csv2ImageDepthDataset(
            csv_file=val_csv_path,
            root_dir=self.root_dir
        )
        self.val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.random.HorizontalFlip(),
        ])
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          transform=self.val_transform)

    def test_dataloader(self):
        test_csv_path = os.path.join(self.csv_dir, 'test.csv')
        self.test_dataset = Csv2ImageDepthDataset(
            csv_file=test_csv_path,
            root_dir=self.root_dir
        )
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
