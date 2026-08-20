# Lab 4: MNIST Classification (MLP & CNN)

Classifies handwritten digits from the MNIST dataset using two different architectures — a multilayer fully connected network (MLP) and a simple convolutional network (CNN) — and compares three optimizers.

## Contents (`lab4_mlp_cnn_mnist.ipynb`)

### Models
- **MLP** — `Flatten → 784→512→256→128→10`, ReLU activations, Dropout(0.2) between hidden layers
- **SimpleCNN** — two `Conv2d → ReLU → MaxPool2d` blocks (1→32→64 channels) followed by a fully connected classifier head
- (A LeNet variant is also defined for comparison)

### Training Setup
- **Dataset:** MNIST (auto-downloaded via `torchvision.datasets.MNIST`), normalized with mean/std `(0.1307, 0.3081)`
- **Epochs:** 10, **Batch size:** 128, **Learning rate:** 0.001
- **Loss:** CrossEntropyLoss
- **Optimizers compared:**
  - SGD + Momentum (0.9) + weight decay (5e-4)
  - Adam + weight decay (1e-4)
  - AdamW + weight decay (1e-4)

A shared `train_model()` function trains and evaluates a given (model, optimizer) pair per epoch, tracking training/test loss and accuracy, and reports the best test accuracy achieved.

## Tech Stack
- Python, PyTorch, torchvision

## Requirements
```bash
pip install torch torchvision
```
MNIST is downloaded automatically on first run (`root='./data'`).

## Usage
Run all cells in `lab4_mlp_cnn_mnist.ipynb` sequentially. GPU is used automatically if available (`torch.cuda.is_available()`).

## Notes
- The notebook trains and compares both models across all three optimizers, to evaluate the effect of optimizer choice and regularization (dropout, weight decay) on convergence and generalization.