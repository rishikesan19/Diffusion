"""Dataset loading utilities for diffusion models."""

from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Tuple , Optional
from torchvision import transforms
import os

from diffusion_models.datasets.data_utils import get_preprocess_transform, transform
from diffusion_models.datasets.attribute_dataset import AttributeDataset


def setup_dataloader(
    data_source: str,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
    split: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, transforms.Compose]:
    """Setup a dataset with preprocessing.
    
    Args:
        data_source: Either:
            - A string path to a local image folder, or
            - A string containing the Hugging Face dataset name
        batch_size: Batch size for the dataloader
        image_size: Target size for the images
        shuffle: Whether to shuffle the data
        split: Dataset split to load
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster data transfer to GPU
        drop_last: Whether to drop the last incomplete batch

    Returns:
        Tuple containing:
            - Dataloader
            - Preprocessing transform
    """
    # Setup preprocessing
    preprocess = get_preprocess_transform(image_size)
    
    # Check if data source exists locally
    if os.path.exists(data_source):
        # Local image folder
        dataset = load_dataset("imagefolder", data_dir=data_source, split=split)
        print(f"Loaded local dataset from: {data_source}")
        print(f"Number of images in the dataset: {len(dataset)}")
        
        # For imagefolder datasets, the image column is named 'image'
        image_column = 'image'
    else:
        # Try to load from Hugging Face
        try:
            dataset = load_dataset(data_source, split=split)
            print(f"Loaded Hugging Face dataset: {data_source}")
            print(f"Number of images in the dataset: {len(dataset)}")
            
            # Try to find the image column
            possible_image_columns = ['image', 'img', 'images', 'pixel_values']
            image_column = next((col for col in possible_image_columns if col in dataset.column_names), None)
            
            if image_column is None:
                raise ValueError(
                    f"Could not find image column in dataset. Available columns: {dataset.column_names}"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from {data_source}. "
                f"Please check if the path exists locally or if the Hugging Face dataset name is correct. "
                f"Error: {str(e)}"
            )
    
    # Apply transforms
    dataset.set_transform(lambda examples: transform(examples, preprocess))
    
    # Create dataloader with all specified settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader, preprocess


def create_attribute_dataloader(
    image_dir: str,
    attribute_label_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: int = 256,
    segmentation_dir: Optional[str] = None
) -> DataLoader:
    """Create a DataLoader for the attribute dataset.
    
    Args:
        image_dir (str): Directory containing the image files
        attribute_label_path (str): Path to the attribute label file
        batch_size (int): Batch size for the dataloader
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        image_size (int): Size to resize images to (both height and width)
        segmentation_dir (Optional[str]): Directory containing segmentation masks
        
    Returns:
        DataLoader: DataLoader for the attribute dataset
    """
    preprocess = get_preprocess_transform(image_size)

    # Create the dataset
    dataset = AttributeDataset(
        image_dir=image_dir,
        attribute_label_path=attribute_label_path,
        segmentation_dir=segmentation_dir,
        image_size=image_size,
        transform=preprocess
    )
    
    # Create and return the dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataloader creation
    try:
        # Test local image folder
        print("\nTesting local image folder loader:")
        local_loader, preprocess = setup_dataloader(
            data_source="data/CelebA-HQ-split/test_300",
            batch_size=8,
            image_size=256,
            num_workers=2
        )
        batch = next(iter(local_loader))
        print(f"Local batch shape: {batch['images'].shape}")
        
        # Test Hugging Face dataset
        print("\nTesting Hugging Face dataset loader:")
        hf_loader, preprocess = setup_dataloader(
            data_source="huggan/smithsonian_butterflies_subset",  # Example with Smithsonian butterflies dataset
            batch_size=8,
            image_size=256,
            num_workers=2
        )
        batch = next(iter(hf_loader))
        print(f"HF batch shape: {batch['images'].shape}")
        
        # Test attribute dataloader
        print("\nTesting attribute dataloader:")
        train_loader = create_attribute_dataloader(
            image_dir="data/CelebA-HQ-split/test_300",
            attribute_label_path="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt",
        )
        batch = next(iter(train_loader))
        images, attributes = batch
        print(f"Attribute batch shapes - Images: {images.shape}, Attributes: {attributes.shape}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
