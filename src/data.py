from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
import torch
from PIL import Image

class CroppedImageDataset(Dataset):
    def __init__(self, data_path="", transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(data_path, "*.png")))
        self.class_to_idx = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Extract the label from the filename
        label_name = os.path.basename(image_path).split('_')[0]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.int64)