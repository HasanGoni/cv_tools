#!/usr/bin/env python3
"""
Extract a single image from the Electron Microscopy Dataset
Dataset: https://huggingface.co/datasets/hasangoni/Electron_microscopy_dataset
"""

from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def extract_single_image(index=0, split='train', save_path=None):
    """
    Extract a single image from the electron microscopy dataset
    
    Args:
        index: Index of the image to extract (default: 0)
        split: Dataset split to use ('train' or 'test', default: 'train')
        save_path: Path to save the image (optional)
    
    Returns:
        tuple: (image_array, label_array) - both as numpy arrays
    """
    
    # Load the dataset
    print(f"Loading dataset from Hugging Face...")
    dataset = load_dataset("hasangoni/Electron_microscopy_dataset")
    
    # Get the specific split
    data_split = dataset[split]
    
    print(f"Dataset info:")
    print(f"  - Total samples in {split}: {len(data_split)}")
    print(f"  - Available splits: {list(dataset.keys())}")
    
    # Extract the image at the specified index
    if index >= len(data_split):
        raise ValueError(f"Index {index} is out of range. Dataset has {len(data_split)} samples.")
    
    sample = data_split[index]
    
    # Get image and label
    image = sample['image']  # PIL Image
    label = sample['label']  # PIL Image (mask)
    
    # Convert to numpy arrays
    image_array = np.array(image)
    label_array = np.array(label)
    
    print(f"Extracted image {index}:")
    print(f"  - Image shape: {image_array.shape}")
    print(f"  - Image dtype: {image_array.dtype}")
    print(f"  - Label shape: {label_array.shape}")
    print(f"  - Label dtype: {label_array.dtype}")
    print(f"  - Label unique values: {np.unique(label_array)}")
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        image.save(save_path / f"image_{index}.png")
        label.save(save_path / f"label_{index}.png")
        
        print(f"Images saved to: {save_path}")
    
    return image_array, label_array

def visualize_sample(image_array, label_array, index=0):
    """Visualize the image and its corresponding label mask"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title(f'Original Image (Index: {index})')
    axes[0].axis('off')
    
    # Label mask
    axes[1].imshow(label_array, cmap='gray')
    axes[1].set_title('Label Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_array, cmap='gray', alpha=0.7)
    axes[2].imshow(label_array, cmap='Reds', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to demonstrate usage"""
    
    # Extract first image from train split
    print("Extracting first image from train split...")
    image, label = extract_single_image(index=0, split='train', save_path='extracted_images')
    
    # Visualize
    visualize_sample(image, label, index=0)
    
    # Extract a different image
    print("\nExtracting 10th image from train split...")
    image2, label2 = extract_single_image(index=10, split='train')
    visualize_sample(image2, label2, index=10)
    
    # Extract from test split
    print("\nExtracting first image from test split...")
    image3, label3 = extract_single_image(index=0, split='test')
    visualize_sample(image3, label3, index=0)

if __name__ == "__main__":
    main() 