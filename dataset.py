import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class VOCDatasetManager:
    def __init__(self, config):
        self.config = config

        self.img_transform = transforms.Compose([
            transforms.Resize(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(config["img_size"], interpolation=transforms.InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.int64))
        ])

    def get_dataloaders(self):
        train_ds = datasets.VOCSegmentation(
            root=self.config["data_path"], year='2012', image_set='train', download=True,
            transform=self.img_transform, target_transform=self.mask_transform
        )
        test_ds = datasets.VOCSegmentation(
            root=self.config["data_path"], year='2012', image_set='val', download=True,
            transform=self.img_transform, target_transform=self.mask_transform
        )

        return (DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True),
                DataLoader(test_ds, batch_size=self.config["batch_size"], shuffle=False))