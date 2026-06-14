import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF


class JointTransform:
    def __init__(self, config, is_train):
        self.img_size = config["img_size"]
        self.is_train = is_train
        self.apply_epic = config.get("apply_epic", False)

    def __call__(self, img, mask):
        if self.is_train and not self.apply_epic:
            scale = random.uniform(0.5, 2.0)
            new_h = int(self.img_size[0] * scale)
            new_w = int(self.img_size[1] * scale)

            img = TF.resize(img, [new_h, new_w])
            mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

            pad_w = max(self.img_size[1] - img.size[0], 0)
            pad_h = max(self.img_size[0] - img.size[1], 0)

            if pad_w > 0 or pad_h > 0:
                img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
                mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)

            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.img_size)
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img = TF.resize(img, self.img_size)
            mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask


class VOCDatasetManager:
    def __init__(self, config):
        self.config = config

    def get_dataloaders(self):
        train_ds = datasets.VOCSegmentation(
            root=self.config["data_path"], year='2012', image_set='train', download=True,
            transforms=JointTransform(self.config, is_train=True)
        )
        test_ds = datasets.VOCSegmentation(
            root=self.config["data_path"], year='2012', image_set='val', download=True,
            transforms=JointTransform(self.config, is_train=False)
        )

        return (DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True),
                DataLoader(test_ds, batch_size=self.config["batch_size"], shuffle=False))