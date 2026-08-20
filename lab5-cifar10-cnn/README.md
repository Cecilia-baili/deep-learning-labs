# Lab 5: CIFAR-10 Classification (Custom CNN)

Designs a custom convolutional neural network (in the style of VGG/GoogLeNet/ResNet, built from scratch rather than using `torchvision.models`) to classify CIFAR-10 images, with data augmentation, checkpointing, and a cosine-annealing learning rate schedule.

## Contents (`lab5_cnn_cifar10.ipynb`)

### Data Pipeline
- **Dataset:** CIFAR-10 (auto-downloaded via `torchvision.datasets.CIFAR10`)
- **Augmentation:** random crop, random horizontal flip, and per-channel normalization (mean/std computed via a custom `get_mean_and_std()` helper)
- **Batch size:** 128

### Model — `YourNet`
A custom CNN stacking `Conv2d → BatchNorm2d → ReLU` blocks with `MaxPool2d` downsampling, designed manually rather than imported from a pretrained architecture.

### Training Setup
- **Epochs:** 200
- **Learning rate:** 0.001, with `CosineAnnealingLR` scheduling over the full training run
- **Optimizer:** SGD + Momentum (0.9) + weight decay (5e-4)
- **Loss:** CrossEntropyLoss
- **Checkpointing:** the best model (by test accuracy) is saved to `best_model.pth`; training automatically resumes from a saved checkpoint if one is found, restoring the epoch counter and best accuracy

## Tech Stack
- Python, PyTorch, torchvision

## Requirements
```bash
pip install torch torchvision
```
CIFAR-10 is downloaded automatically on first run (`root='/data/cifar10'` — update this path if needed).

## Usage
Run all cells in `lab5_cnn_cifar10.ipynb` sequentially. GPU is used automatically if available. Training can be safely interrupted and resumed thanks to the checkpoint-loading logic.

## Notes
- The 200-epoch training run with cosine annealing is computationally intensive — a GPU is strongly recommended.