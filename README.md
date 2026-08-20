# Deep Learning Labs

Four lab assignments covering core deep learning architectures — fully connected networks, CNNs, Transformers, and reinforcement learning — implemented in PyTorch.

## Labs

### [Lab 4: MNIST Classification (MLP & CNN)](./lab4-mnist-classification)
Classifies handwritten digits (MNIST) using a multilayer fully connected network (MLP) and a simple CNN, comparing SGD+Momentum, Adam, and AdamW optimizers.

### [Lab 5: CIFAR-10 Classification (Custom CNN)](./lab5-cifar10-cnn)
Designs a custom convolutional network (VGG/GoogLeNet/ResNet-style, built from scratch rather than imported) for CIFAR-10 classification, with data augmentation, checkpoint resuming, and a cosine-annealing LR schedule.

### [Lab 6: Transformer for Text Classification](./lab6-transformer-text-classification)
Implements a Transformer encoder (positional encoding, multi-head attention, feed-forward blocks) from scratch for text classification on the AG News dataset, using a BERT tokenizer for text preprocessing.

### [Lab 7: Double DQN (Reinforcement Learning)](./lab7-double-dqn)
Trains a Double/Dueling DQN agent to solve CartPole-v1 with Gymnasium, using experience replay and a target network.

## Tech Stack
- Python, PyTorch, torchvision
- Hugging Face `transformers` / `datasets` (Lab 6)
- Gymnasium (Lab 7)
- Matplotlib, TensorBoard (visualization/logging)

## Notes
Some notebooks contain instructor-provided fill-in-the-blank sections (`## 完成此处代码` / `Your code`) that were completed as part of the assignment.