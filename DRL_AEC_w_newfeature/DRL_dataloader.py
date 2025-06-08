import io

import torch

from manifold.clients.python import ManifoldClient
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class SceneDataset(Dataset):
    def __init__(
        self, manifold_path, angle, transform=None, crop_size=128, black_margin_size=100
    ):
        self.bucket, self.path = manifold_path.split("/", 1)
        self.client = ManifoldClient(self.bucket)
        self.file_list = [f[0] for f in self.client.sync_ls(self.path)]
        self.angle = angle
        self.transform = transform
        self.crop_size = crop_size
        self.black_margin_size = black_margin_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        try:
            idx_part = file_name.split(f"205-{self.angle}-")[1].split("-")[0]
            idx_value = int(idx_part)
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract idx from file name: {file_name}")
            print(f"Error: {e}")
            return None
        file_path = f"{self.path}/{file_name}"
        stream = io.BytesIO()
        self.client.sync_get(file_path, stream)
        image = Image.open(stream).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Apply dataset augmentation
        image = self.image_augmentation(image)
        return {"image": image, "idx": idx_value} 

    def image_augmentation(self, image):
        # Ensure the crop does not start or end within the black margin
        # Check if the image is a PIL Image or a tensor
        if isinstance(image, Image.Image):
            width, height = image.size
        elif torch.is_tensor(image):
            # Assuming the image tensor is in CxHxW format
            _, height, width = image.shape
        else:
            raise TypeError("Unsupported image type")
        top = self.black_margin_size
        left = self.black_margin_size
        right = width - self.black_margin_size - self.crop_size
        bottom = height - self.black_margin_size - self.crop_size
        # Randomly choose the top-left corner of the crop
        i = torch.randint(low=top, high=bottom + 1, size=(1,)).item()
        j = torch.randint(low=left, high=right + 1, size=(1,)).item()
        # Crop the image
        image = F.crop(image, i, j, self.crop_size, self.crop_size)
        # Random horizontal and vertical flips
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
        if torch.rand(1) < 0.5:
            image = F.vflip(image)
        return image
