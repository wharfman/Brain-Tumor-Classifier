# Brain Tumor Predictor

Classifies brain MRI scans into four classes: **glioma**, **meningioma**, **pituitary tumor** or **no tumor**. It fine-tunes an Ultralytics YOLO11 classifier (PyTorch) with three added modules (CBAM attention, SPPF+ pooling and a BiFPN head). A local web dashboard lets you upload a scan and see the prediction.

## Setup

1. Install PyTorch for your hardware **first**. Otherwise Ultralytics installs the CPU-only build.
   - NVIDIA: use the command from <https://pytorch.org/get-started/locally/>
   - AMD (ROCm on Windows): follow AMD's ROCm PyTorch instructions
2. Install the project and its dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Put the raw images in one folder per class:
   ```
   data/Train/{glioma,meningioma,notumor,pituitary}/
   data/Test/{glioma,meningioma,notumor,pituitary}/
   ```

## Usage

Run every command from the project root.

```
python -m src.preprocess                  # build the processed dataset (train/val/test)
python -m src.train --task classify       # train and save models/classify_best.pt
python -m src.evaluate --task classify    # accuracy and confusion matrix on the test split
python -m src.predict path/to/scan.jpg    # predict one image or a folder
python -m src.dashboard                   # web dashboard at http://127.0.0.1:8000
```

Useful options:

| Command | Option | Effect |
|---|---|---|
| `train` | `--epochs N --batch N` | Training length and batch size (defaults: 50, 32) |
| `train` | `--weights yolo11m-cls.pt` | Start from a different pretrained model |
| `train` | `--device 0` / `--device cpu` | Choose the GPU or force the CPU |
| `preprocess` | `--overwrite` | Rebuild the processed dataset from scratch |
| `dashboard` | `--port N --weights path.pt` | Change the port or the model file |

## How it works

1. **Preprocessing.** Every scan goes through the same pipeline: grayscale, crop to the head, resize to 640×640 with padding, blur, sharpen, then erosion and dilation. Prediction uses the same pipeline, so training and inference inputs match. `Train` is split 85/15 into train and validation, and `Test` is kept for the final evaluation. The training split also gets synthetic copies (rotations, flips, zoom, brightness and noise). Validation and test images are never augmented.
2. **Model.** The model starts from pretrained `yolo11s-cls` weights and adds three modules:
   - **CBAM** (channel and spatial attention) after backbone layers 4, 6 and 8
   - **SPPF+** (multi-scale pooling) in front of layer 9
   - **BiFPN** (weighted multi-scale fusion), which replaces the standard classification head
3. **Training.** Ultralytics writes each run to `runs/classify/<name>/`, and the best checkpoint is copied to `models/classify_best.pt`. The evaluate, predict and dashboard commands load that file by default.

The settings for all of the above (paths, image size, augmentation, which layers get each module, training hyperparameters) are in `src/config.py`.

## Project structure

```
data/                  raw images (Train/, Test/) and the generated processed/ dataset
models/                best trained weights (classify_best.pt)
runs/                  full training outputs: checkpoints, metrics, plots
src/
  config.py            all settings and paths
  transforms.py        image preprocessing pipeline
  augment.py           synthetic training copies
  preprocess.py        builds the train/val/test dataset
  dataloader.py        locates and validates the dataset
  model.py             loads models and weights
  cbam.py              CBAM attention module
  sppf_plus.py         SPPF+ pooling module
  bifpn.py             BiFPN classification head
  trainer.py           trainer that adds the modules above
  progress.py          per-epoch console output during training
  train.py             training entry point
  evaluate.py          test-set evaluation
  predict.py           predicts on new images
  dashboard.py/.html   local upload dashboard
tests/                 pytest suite
```

