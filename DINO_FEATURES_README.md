# DINO Features for Perceptual Loss

This document explains how to use the new DINO (Distillation of No Labels) feature extraction system for perceptual loss computation in the Gaussian Splatting training pipeline.

## Overview

We've replaced the on-the-fly VGG feature extraction with offline DINO feature extraction to improve training performance and provide better perceptual loss computation. DINO features are more semantically meaningful and provide better supervision for neural rendering tasks.

## Changes Made

### 1. New Files Created

- `utils/dino_utils.py`: Contains DINO feature extractor and utilities
- `extract_dino_features.py`: Script to extract DINO features offline
- `DINO_FEATURES_README.md`: This documentation file

### 2. Modified Files

- `utils/loss_utils.py`: Replaced VGG with DINO feature extractor
- `train.py`: Updated to use pre-extracted DINO features
- `arguments/__init__.py`: Added `dino_feature_dir` parameter

## Usage Instructions

### Step 1: Extract DINO Features Offline

Before training, you need to extract DINO features for all your training images:

```bash
python extract_dino_features.py \
    --image_dir /path/to/your/training/images \
    --output_dir /path/to/save/dino_features
```

**Example:**
```bash
python extract_dino_features.py \
    --image_dir data/EndoNeRF/cutting/cutting_tissues_twice/images \
    --output_dir data/EndoNeRF/cutting/cutting_tissues_twice/dino_features
```

This will:
- Load all images from the specified directory
- Extract DINO features using a pre-trained Vision Transformer
- Save features as `.npy` files with the same base names as the images

### Step 2: Train with DINO Features

When running training, specify the DINO feature directory:

```bash
python train.py \
    --source_path /path/to/your/dataset \
    --model_path /path/to/save/model \
    --dino_feature_dir /path/to/dino_features \
    [other training arguments...]
```

**Example:**
```bash
python train.py \
    --source_path data/EndoNeRF/cutting/cutting_tissues_twice \
    --model_path output/cutting_dino \
    --dino_feature_dir data/EndoNeRF/cutting/cutting_tissues_twice/dino_features \
    --iterations 50000 \
    --lambda_perceptual 1.0
```

## How It Works

### Feature Extraction
- Uses a pre-trained Vision Transformer (ViT-Base) from the `timm` library
- Extracts global features from 224x224 resized images
- Features are normalized and saved as numpy arrays

### Perceptual Loss Computation
- During training, predicted images have DINO features extracted on-the-fly
- Ground truth images use pre-extracted features (loaded from disk)
- Loss is computed using cosine similarity between feature vectors
- Falls back to on-the-fly extraction if pre-extracted features are not found

### Performance Benefits
- **Faster Training**: No need to extract features for ground truth images during training
- **Better Features**: DINO features are more semantically meaningful than VGG features
- **Memory Efficient**: Features are loaded only when needed

## Configuration

### DINO Model
The system uses `vit_base_patch16_224` by default. You can modify this in `utils/dino_utils.py`:

```python
class DINOFeatureExtractor:
    def __init__(self, model_name='vit_base_patch16_224', device='cuda'):
        # Change model_name to use different architectures
```

### Perceptual Loss Weight
Control the strength of perceptual loss using the `--lambda_perceptual` parameter:

```bash
--lambda_perceptual 1.0  # Default value
```

## Troubleshooting

### Missing DINO Features
If you get a `FileNotFoundError` for DINO features:
1. Make sure you've run the feature extraction script first
2. Check that the `--dino_feature_dir` path is correct
3. Verify that feature files exist with `.npy` extensions

### Memory Issues
If you encounter memory issues:
1. Reduce batch size in feature extraction (modify `extract_dino_features.py`)
2. Use a smaller DINO model (e.g., `vit_small_patch16_224`)
3. Process images in smaller batches

### Performance Optimization
For faster feature extraction:
1. Use multiple workers: modify `extract_dino_features.py` to use `DataLoader` with `num_workers`
2. Use GPU acceleration: ensure CUDA is available
3. Process images in batches: modify the extraction script for batch processing

## File Structure

After running the feature extraction, your directory structure should look like:

```
data/EndoNeRF/cutting/cutting_tissues_twice/
├── images/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── dino_features/
│   ├── 00000.npy
│   ├── 00001.npy
│   └── ...
└── [other dataset files]
```

## Dependencies

Make sure you have the required dependencies:

```bash
pip install timm  # For DINO model
pip install torch torchvision  # PyTorch
pip install numpy  # For array operations
pip install tqdm  # For progress bars
```

## Notes

- DINO features are more robust to domain shifts and provide better semantic supervision
- The feature extraction is a one-time process per dataset
- You can reuse the same features for multiple training runs
- The system gracefully falls back to on-the-fly extraction if pre-extracted features are missing 