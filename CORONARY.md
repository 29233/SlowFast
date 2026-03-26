# Coronary CT Angiography Dataset with SlowFast

This document describes how to train and evaluate the MViTV2-B model on the coronary CT angiography dataset using the SlowFast framework.

## Dataset Structure

The coronary CT dataset is organized as follows:

```
Central/
├── metadataV0.csv          # Master metadata file with all samples
├── train.csv               # Training split (77 samples)
├── val.csv                 # Validation split (77 samples)
├── test.csv                # Test split (20 samples)
├── 163329/                 # Patient ID folder
│   ├── IM-0001-0001.dcm    # DICOM frame 1
│   ├── IM-0001-0002.dcm    # DICOM frame 2
│   └── ...
├── 202367/
├── 222773/
└── ...
```

### Metadata Format

The `metadataV0.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| ID | Patient/sample identifier (matches folder name) |
| Start Frame | First usable frame index in the DICOM sequence |
| End Frame | Last usable frame index in the DICOM sequence |
| Plaque | Plaque percentage (regression target) |
| Branch | Coronary artery branch (RCA, LAD, LCX, etc.) |
| SplitA | Train/test assignment (version A) |
| SplitB | Train/test assignment (version B) |

## Configuration

### YAML Configuration File

The configuration file is located at `configs/Coronary/MVITv2_B_32x3.yaml`. Key settings:

```yaml
DATA:
  NUM_FRAMES: 32          # Number of frames to sample
  SAMPLING_RATE: 2        # Frame sampling rate
  TRAIN_CROP_SIZE: 224    # Training crop size
  TEST_CROP_SIZE: 224     # Test crop size
  PATH_TO_DATA_DIR: /path/to/Central  # Update this!
  PATH_PREFIX: /path/to/Central       # Update this!
  SPLIT_COLUMN: SplitA    # Use SplitA or SplitB

MODEL:
  NUM_CLASSES: 1          # Single output for regression
  LOSS_FUNC: mse          # MSE loss for regression

SOLVER:
  BASE_LR: 0.00005        # Learning rate
  MAX_EPOCH: 100          # Training epochs
  BATCH_SIZE: 4           # Batch size per GPU
```

## Setup

### 1. Update Paths

Edit `configs/Coronary/MVITv2_B_32x3.yaml` and update the following paths:

```yaml
DATA:
  PATH_TO_DATA_DIR: /absolute/path/to/DcmDataset/Central
  PATH_PREFIX: /absolute/path/to/DcmDataset/Central
```

### 2. Generate Split CSV Files

Run the split generation script to create train/val/test CSV files:

```bash
cd SlowFast
python tools/generate_coronary_splits.py
```

This will generate:
- `train.csv` - Training samples
- `val.csv` - Validation samples
- `test.csv` - Test samples

## Training

### Single Node Training

To train MViTV2-B on the coronary CT dataset:

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \
  DATA.PATH_TO_DATA_DIR /path/to/Central \
  DATA.PATH_PREFIX /path/to/Central
```

### Multi-GPU Training

For multi-GPU training (if available):

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  NUM_GPUS 4 \
  TRAIN.BATCH_SIZE 4 \
  DATA.PATH_TO_DATA_DIR /path/to/Central \
  DATA.PATH_PREFIX /path/to/Central
```

### Using Different Splits

To use SplitB instead of SplitA:

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  DATA.SPLIT_COLUMN SplitB \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4
```

## Evaluation

To evaluate a trained model on the test set:

```bash
python tools/run_net.py \
  --cfg configs/Coronary/MVITv2_B_32x3.yaml \
  TEST.ENABLE True \
  TEST.CHECKPOINT_PATH /path/to/checkpoint.pkl \
  NUM_GPUS 1
```

## Output

Training outputs are saved to `OUTPUT_DIR` (default: `./outputs/coronary_mvitt2_b/`):

```
outputs/coronary_mvitt2_b/
├── config.yaml           # Training configuration
├── log.txt              # Training log
├── checkpoint_epoch_XXX.pkl  # Model checkpoints
└── runs/                # TensorBoard logs
```

## Key Implementation Details

### DICOM Reading

The dataset class uses `dcm.py` utilities for DICOM reading:
- Automatic series sorting by `ImagePositionPatient`
- HU value conversion using `RescaleSlope` and `RescaleIntercept`
- Cardiac window/level preset (width: 400, center: 50)

### Frame Sampling

1. Load full DICOM sequence from patient folder
2. Extract frames within `[Start Frame, End Frame]` range
3. Apply cardiac window/level for visualization
4. Sample `NUM_FRAMES` frames with `SAMPLING_RATE`
5. Random temporal sampling for training
6. Center temporal sampling for testing

### Data Augmentation

- Random horizontal flip (training only)
- Random spatial cropping and scaling
- No color jitter (CT images are grayscale)

### Regression Target

The Plaque column contains continuous values (0-100) representing plaque percentage. The model outputs a single scalar for regression.

## Troubleshooting

### Memory Issues

If you encounter OOM errors:
- Reduce `BATCH_SIZE`
- Reduce `NUM_FRAMES`
- Use gradient accumulation

### Slow Loading

If data loading is slow:
- Increase `DATA_LOADER.NUM_WORKERS`
- Enable `DATA_LOADER.PIN_MEMORY: True`
- Ensure DICOM files are on SSD

### Missing DICOM Files

If some samples fail to load:
- Check that patient ID folder exists
- Verify DICOM files are present
- Check `Start Frame` and `End Frame` are valid

## Citation

If you use this code or dataset, please cite:

```
@article{slowfast,
  title={SlowFast Networks for Video Recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  journal={ICCV},
  year={2019}
}

@article{mvitv2,
  title={Multiscale Vision Transformers},
  author={Fan, Haoqi et al.},
  journal={ICCV},
  year={2021}
}
```
