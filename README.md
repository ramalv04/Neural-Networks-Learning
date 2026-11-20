# Neural Networks Learning


This repository is my learning path for neural networks and Deep Learning. I am using it to collect small projects as I progress from the fundamentals to more advanced architectures.

Each project focuses on a specific idea, dataset, or training workflow. The goal is not only to build models, but also to understand what is happening inside them and how the concepts connect across different implementations.

## Projects

### 01. XOR from Scratch

[Project README](01-XOR-From-Scratch/README.md)

My first project was a neural network built completely from scratch with NumPy. It solves the XOR problem and shows the core ideas behind forward propagation, loss calculation, backpropagation, gradient descent, and parameter updates without using a Deep Learning framework.

### 02. MNIST MLP with PyTorch

[Project README](02-MNIST-MLP/README.md)

The second project moves to PyTorch and trains a Multilayer Perceptron on MNIST. It introduces tensors, DataLoader, mini-batches, ReLU, logits, CrossEntropy loss, SGD, and automatic differentiation with Autograd.

### 03. MNIST CNN

[Project README](03-MNIST-CNN/README.md)

The third project builds a Convolutional Neural Network for MNIST. It extends the MLP idea from project 2 with convolution and pooling layers, adds filter and activation visualizations, and shows why CNNs are a better fit for image data than a purely fully connected network.

### 04. Transfer Learning with ResNet18

[Project README](04-transfer-learning/README.md)

The fourth project applies transfer learning with a pretrained ResNet18 on the Oxford-IIIT Pet dataset. It replaces the final classification layer, freezes most pretrained layers, fine-tunes Layer4 and the classifier, and uses augmentation, ImageNet normalization, and visualizations such as confusion matrices, top-5 predictions, filters, and activation maps.

### 05. Convolutional Autoencoder

[Project README](05-convolutional-autoencoder/README.md)

This project explores image reconstruction with convolutional autoencoders and compares a basic autoencoder with a convolutional architecture.

## Neural Networks Course

[Course folder](Materia-Redes-Neuronales/README.md)

The `Materia-Redes-Neuronales` folder collects the practical assignments completed for the university Neural Networks course. It includes a numerical simulation of the Izhikevich neuron model and a feedforward neural network trained to classify Fashion-MNIST. The notebooks contain the implementations, and the PDFs contain the complete scientific reports.

## What this repository is for

- Practicing neural network fundamentals step by step.
- Comparing manual implementations with framework-based training.
- Building intuition around training loops, optimization, and evaluation.
- Keeping a clear record of what I learn in each stage.

## Technologies used so far

- Python
- NumPy
- PyTorch
- TorchVision
- Matplotlib
- Scikit-learn

## Repository structure

```text
01-XOR-From-Scratch/
02-MNIST-MLP/
03-MNIST-CNN/
04-transfer-learning/
05-convolutional-autoencoder/
Materia-Redes-Neuronales/
```

---

If you want the Spanish version of this overview, open [README_ES.md](README_ES.md).

