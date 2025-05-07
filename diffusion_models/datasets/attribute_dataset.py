import os
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class AttributeDataset(Dataset):
    """Dataset class for loading images with attribute labels.
    
    This dataset loads images from a directory and their corresponding attribute labels
    from a CSV-formatted text file. The attribute labels are binary (-1 for no, 1 for yes).
    
    Args:
        image_dir (str): Directory containing the image files
        attribute_label_path (str): Path to the attribute label file
        image_size (int): Size to resize images to (both height and width)
        transform (Optional[transforms.Compose]): Optional transforms to apply to images
        segmentation_dir (Optional[str]): Path to the segmentation masks (grayscale part-wise masks)
    """

    def __init__(
        self,
        image_dir: str,
        attribute_label_path: str,
        segmentation_dir: Optional[str] = None,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.segmentation_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]) if segmentation_dir else None

        existing_images = set(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
        print(f"Found {len(existing_images)} images in directory")

        print(f"Reading attribute file: {attribute_label_path}")

        with open(attribute_label_path, 'r') as f:
            f.readline()  # Skip first line
            header_line = f.readline().strip()
            attribute_names = header_line.split()

        self.attributes_df = pd.read_csv(
            attribute_label_path,
            skiprows=2,
            sep=r'\s+',
            header=None,
            names=['image_id'] + attribute_names,
            dtype=str
        )

        self.attributes_df['image_id'] = self.attributes_df['image_id'].apply(
            lambda x: f"{x}.jpg" if not x.endswith('.jpg') else x
        )

        for col in self.attributes_df.columns[1:]:
            self.attributes_df[col] = pd.to_numeric(self.attributes_df[col], errors='coerce')
            self.attributes_df[col] = self.attributes_df[col].map({-1: 0, 1: 1})

        self.attributes_df = self.attributes_df[self.attributes_df['image_id'].isin(existing_images)]

        if len(self.attributes_df) == 0:
            print("Debugging information:")
            print(f"Total images in attribute file: {len(self.attributes_df)}")
            print("Sample of image_ids in attribute file before filtering:")
            print(self.attributes_df['image_id'].head() if len(self.attributes_df) > 0 else "No images found")
            print("Sample of image_ids in directory:")
            print(list(existing_images)[:5])
            raise ValueError(
                f"No matching images found between {image_dir} and {attribute_label_path}\n"
                f"Found {len(existing_images)} images in directory\n"
                f"First few images in directory: {list(existing_images)[:5]}\n"
                f"First few images in attribute file: {list(self.attributes_df['image_id'])[:5] if len(self.attributes_df) > 0 else 'No images found'}"
            )

        self.attribute_names = self.attributes_df.columns[1:].tolist()
        print(f"Final dataset size: {len(self.attributes_df)} images with attributes out of {len(existing_images)} images in directory")

    def __len__(self) -> int:
        return len(self.attributes_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.attributes_df.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        attributes = torch.from_numpy(row[1:].values.astype(np.float32))
        sample = {
            "image": image,
            "attribute": attributes
        }

        if self.segmentation_dir:
            base_id = image_id.replace(".jpg", "")
            prefix = f"{int(base_id):05d}"
            mask_files = [f for f in os.listdir(self.segmentation_dir) if f.startswith(prefix)]

            combined_mask = None
            for mask_file in mask_files:
                mask_path = os.path.join(self.segmentation_dir, mask_file)
                mask = Image.open(mask_path).convert("L")
                mask_tensor = self.segmentation_transform(mask)

                if combined_mask is None:
                    combined_mask = mask_tensor
                else:
                    combined_mask = torch.maximum(combined_mask, mask_tensor)

            if combined_mask is not None:
                sample["segmentation"] = combined_mask

        return sample

    def get_attribute_names(self) -> List[str]:
        return self.attribute_names
