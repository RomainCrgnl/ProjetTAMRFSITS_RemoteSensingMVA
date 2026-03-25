#!/usr/bin/env python3
"""
Change Detection Runner Script

This script loads the change detection functions and runs them with
command-line provided experiment paths.

Usage:
    python run_change_detection.py <experiment_path> [output_base]

Example:
    python run_change_detection.py "30SWH_24_c5_g1/predictions/30SWH_24/hr_mae_CUSTOM_FORECAST_50_318.0" "30SWH_24_c5_g1/output_unet"
"""

import sys
import os
from pathlib import Path

# Model configuration
SRCDIR = "."
MODEL_PATH = "fresunet3_final.pth.tar"

# Import all required libraries
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import rasterio
from PIL import Image
import imageio
import glob

# Import the network architecture
sys.path.append(SRCDIR)
from fresunet import FresUNet


def tif_to_rgb(tif_path, red_b=3, green_b=2, blue_b=1):
    """Convert Sentinel-2 TIFF to RGB array with proper scaling and contrast stretch."""
    with rasterio.open(tif_path) as src:
        r = src.read(red_b).astype(np.float32)
        g = src.read(green_b).astype(np.float32)
        b = src.read(blue_b).astype(np.float32)

    rgb = np.dstack((r, g, b))
    rgb[rgb == -10000] = np.nan
    rgb = rgb / 10000.0
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_clipped = np.clip(rgb / 0.3, 0, 1)
    rgb_8bit = (rgb_clipped * 255).astype(np.uint8)

    return rgb_8bit


def normalize_image(img):
    """Normalize image using mean and standard deviation."""
    img_float = img.astype('float')
    return (img_float - img_float.mean()) / img_float.std()


def reshape_for_torch(img):
    """Transpose image from (H, W, C) to (C, H, W) for PyTorch."""
    return torch.from_numpy(img.transpose((2, 0, 1)))


def compute_change_map(tif_path1, tif_path2, net):
    """Compute change detection map between two satellite images."""
    # Convert TIFFs to RGB
    img1_rgb = tif_to_rgb(tif_path1)
    img2_rgb = tif_to_rgb(tif_path2)

    # Normalize images
    img1_norm = normalize_image(img1_rgb)
    img2_norm = normalize_image(img2_rgb)

    # Ensure same dimensions
    s1 = img1_norm.shape
    s2 = img2_norm.shape
    if s1 != s2:
        img2_norm = np.pad(
            img2_norm,
            ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)),
            mode='edge'
        )

    # Convert to PyTorch tensors
    tensor1 = reshape_for_torch(img1_norm)
    tensor2 = reshape_for_torch(img2_norm)

    # Add batch dimension
    tensor1 = Variable(torch.unsqueeze(tensor1, 0).float())
    tensor2 = Variable(torch.unsqueeze(tensor2, 0).float())

    # Run inference
    with torch.no_grad():
        output = net(tensor1, tensor2)

    # Get predictions
    _, predicted = torch.max(output.data, 1)
    change_map = (255 * predicted[0, :, :]).numpy().astype(np.uint8)

    return change_map


def process_experiment(experiment_path, output_base, net):
    """Process all image pairs in an experiment directory."""
    # Create output directories
    pred_output_dir = os.path.join(output_base, "prediction")
    temporal_output_dir = os.path.join(output_base, "temporal_difference")

    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(temporal_output_dir, exist_ok=True)

    # Find all reference TIFF files
    ref_pattern = os.path.join(experiment_path, "*_ref.tif")
    ref_files = glob.glob(ref_pattern)

    if not ref_files:
        print(f"⚠ No reference files found matching: {ref_pattern}")
        return

    # Sort files numerically by prefix
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        return int(filename.split('_')[0])

    ref_files.sort(key=extract_number)

    print(f"Found {len(ref_files)} reference images")
    print(f"Output directory: {output_base}")
    print("=" * 70)

    # Process each image pair
    for i, ref_path in enumerate(ref_files):
        pred_path = ref_path.replace('_ref.tif', '_pred.tif')

        if not os.path.exists(pred_path):
            print(f"⚠ Skipping {os.path.basename(ref_path)} - no matching prediction file")
            continue

        base_name = os.path.basename(ref_path).replace('_ref.tif', '.png')

        print(f"\n[{i+1}/{len(ref_files)}] Processing: {base_name}")
        print(f"  Ref:  {os.path.basename(ref_path)}")
        print(f"  Pred: {os.path.basename(pred_path)}")

        # Compute change map: Reference vs Prediction
        try:
            change_map_pred = compute_change_map(ref_path, pred_path, net)
            pred_output_path = os.path.join(pred_output_dir, base_name)
            imageio.imsave(pred_output_path, change_map_pred)
            print(f"  ✓ Saved prediction change map")
        except Exception as e:
            print(f"  ✗ Error computing prediction change map: {e}")
            continue

        # Compute temporal change map: Reference(t) vs Reference(t-1)
        if i > 0:
            prev_ref_path = ref_files[i-1]
            print(f"  Ref(t-1): {os.path.basename(prev_ref_path)}")

            try:
                change_map_temporal = compute_change_map(ref_path, prev_ref_path, net)
                temporal_output_path = os.path.join(temporal_output_dir, base_name)
                imageio.imsave(temporal_output_path, change_map_temporal)
                print(f"  ✓ Saved temporal change map")
            except Exception as e:
                print(f"  ✗ Error computing temporal change map: {e}")

    print("\n" + "=" * 70)
    print("✓ Processing complete!")
    print(f"\nResults saved to:")
    print(f"  - Prediction changes: {pred_output_dir}")
    print(f"  - Temporal changes:   {temporal_output_dir}")


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) != 2:
        print("Usage: python run_change_detection.py <experiment_path>")
        print("\nExample:")
        print('  python run_change_detection.py "30SWH_24_c5_g1/predictions/30SWH_24/hr_mae_CUSTOM_FORECAST_50_318.0"')
        print("\nOutput will be automatically saved to: <experiment_parent>/output_unet/")
        sys.exit(1)

    experiment_path = sys.argv[1]

    # Auto-generate output path: <experiment_parent>/output_unet
    parent_dir = str(Path(experiment_path).parent.parent.parent)
    output_base = os.path.join(parent_dir, "output_unet")

    print("=" * 70)
    print("Change Detection - UNet Processing")
    print("=" * 70)
    print(f"Experiment path: {experiment_path}")
    print(f"Output path:     {output_base} (auto)")
    print("=" * 70)
    print()

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Error: Model file not found: {MODEL_PATH}")
        print(f"  Make sure you're running from the correct directory")
        sys.exit(1)

    # Load model
    print("Loading pre-trained FresUNet model...")
    net = FresUNet(2*3, 2)
    net.load_state_dict(torch.load(os.path.join(SRCDIR, MODEL_PATH)))
    net.eval()
    print("✓ Model loaded successfully")
    print()

    # Run processing
    process_experiment(experiment_path, output_base, net)


if __name__ == "__main__":
    main()