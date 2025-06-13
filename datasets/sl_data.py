import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    # h = h.movedim(-1, 0)
    h = np.transpose(h, (2, 0, 1))
    return h


class MaskedRelightDataset(Dataset):
    def __init__(
        self,
        image_paths,        # List[str]: paths to input images
        mask_paths,         # List[str]: paths to corresponding binary masks
        bg_paths,           # List[str]: paths to background images
        prompts,            # List[str]: text prompts
        image_width=512,
        image_height=512,
    ):
        assert len(image_paths) == len(mask_paths), "Image and mask list lengths must match."
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.bg_paths = bg_paths
        self.prompts = prompts
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Convert to NumPy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Binary mask: 1 for foreground, 0 for background
        mask_bin = (mask_np > 127).astype(np.uint8)
        mask_3ch = np.stack([mask_bin] * 3, axis=-1)

        # Foreground image with gray (127) background
        fg_np = image_np * mask_3ch + (1 - mask_3ch) * 127
        fg_np = resize_and_center_crop(fg_np, target_width=self.image_width, target_height=self.image_height)
        # fg = numpy2pytorch(fg_np.astype(np.uint8))
        # print(fg.shape)
        # exit(0)

        # Load and resize background image
        bg_path = random.choice(self.bg_paths)
        bg_img = Image.open(bg_path).convert('RGB')
        bg_np = np.array(bg_img)

        bg_np = resize_and_center_crop(bg_np, target_width=self.image_width, target_height=self.image_height)
        # bg = numpy2pytorch(bg_np.astype(np.uint8))

        # Random prompt
        prompt = random.choice(self.prompts)

        return {
            "input_fg": fg_np.astype(np.uint8),
            "input_bg": bg_np.astype(np.uint8),
            "prompt": prompt,
            "path": image_path
        }
