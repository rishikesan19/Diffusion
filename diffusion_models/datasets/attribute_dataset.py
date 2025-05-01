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
        segmentation_dir (Optional[str]): Path to the segmentation masks (RGB images)
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

        # Segmentation-specific transform
        self.segmentation_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]) if segmentation_dir else None
        
        # Get list of existing images in the directory
        existing_images = set(f"{f}" for f in os.listdir(image_dir) if f.endswith('.jpg'))
        print(f"Found {len(existing_images)} images in directory")
        
        # Read the attribute file
        # Skip the first line (number of images) and use the second line as headers
        print(f"Reading attribute file: {attribute_label_path}")
        
        # First read the headers
        with open(attribute_label_path, 'r') as f:
            f.readline()  # Skip first line
            header_line = f.readline().strip()
            attribute_names = header_line.split()
        
        # Now read the data, skipping the first two lines
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
            print(f"Sample of image_ids in attribute file before filtering:")
            print(self.attributes_df['image_id'].head() if len(self.attributes_df) > 0 else "No images found")
            print("Sample of image_ids in directory:")
            print(list(existing_images)[:5])
            
            sample_attr_ids = set(self.attributes_df['image_id'].head().tolist())
            sample_dir_ids = set(list(existing_images)[:5])
            print("Checking for exact matches:")
            print(f"Attribute file IDs: {sample_attr_ids}")
            print(f"Directory IDs: {sample_dir_ids}")
            print(f"Common IDs: {sample_attr_ids.intersection(sample_dir_ids)}")
            
            raise ValueError(
                f"No matching images found between {image_dir} and {attribute_label_path}\n"
                f"Found {len(existing_images)} images in directory\n"
                f"First few images in directory: {list(existing_images)[:5]}\n"
                f"First few images in attribute file: {list(self.attributes_df['image_id'])[:5] if len(self.attributes_df) > 0 else 'No images found'}"
            )

        self.attribute_names = self.attributes_df.columns[1:].tolist()
        print(f"Final dataset size: {len(self.attributes_df)} images with attributes out of {len(existing_images)} images in directory")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.attributes_df)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            dict: {
                'image': image tensor (C, H, W),
                'attribute': attribute tensor (N,),
                'segmentation': segmentation tensor (optional)
            }
        """
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
            seg_path = os.path.join(self.segmentation_dir, image_id)
            segmentation = Image.open(seg_path).convert("RGB")
            segmentation = self.segmentation_transform(segmentation)
            sample["segmentation"] = segmentation

        return sample

    def get_attribute_names(self) -> List[str]:
        """Get the list of attribute names.
        
        Returns:
            List[str]: List of attribute names
        """
        return self.attribute_names
        

# DEBUGGING
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Attributes: 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
    
    try:
        # Initialize the dataset
        dataset = AttributeDataset(
            image_dir="data/CelebA-HQ-split/test_300",
            attribute_label_path="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt"
        )
        
        assert len(dataset) == 300, f"Expected length 300, got {len(dataset)}" 

        # Test attribute names
        attribute_names = dataset.get_attribute_names()
        expected_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
        assert attribute_names == expected_names, f"Expected {expected_names}, got {attribute_names}"
        
        # Test getting an item
        image, attributes = dataset[0]
        assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
        assert isinstance(attributes, torch.Tensor), "Attributes should be a torch.Tensor"
        assert image.shape == (3, 256, 256), f"Expected image shape (3, 256, 256), got {image.shape}"
        assert attributes.shape == (40,), f"Expected attributes shape (40,), got {attributes.shape}"
        
        # Print first image and attributes
        print(f"First image: {image}")
        print(f"First attributes: {attributes}")

        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)
    