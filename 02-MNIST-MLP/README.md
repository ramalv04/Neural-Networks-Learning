[Versión en Español](README_ES.md)

# Project 2 - Handwritten Digit Classification with PyTorch (MNIST)

## Objective

Build and train a Multilayer Perceptron (MLP) using PyTorch to classify handwritten digits from the MNIST dataset.

Unlike the previous project, where every component of the neural network was implemented manually using NumPy, this project focuses on understanding how PyTorch automates the training process through automatic differentiation (Autograd).

---

# Dataset

The model is trained using the **MNIST** dataset.

- 60,000 training images
- 10,000 testing images
- Grayscale images
- Resolution: **28 × 28 pixels**
- 10 classes (digits 0–9)

Each image is transformed into a tensor with shape:

```
1 × 28 × 28
```

Before entering the neural network, the image is flattened into a vector of:

```
784 features
```

---

# Model Architecture

The implemented model is a fully connected neural network.

```
Input (28×28)
↓
Flatten
↓
Linear (784 → 128)
↓
ReLU
↓
Linear (128 → 64)
↓
ReLU
↓
Linear (64 → 10)
↓
Logits
```

Unlike the first project, the final layer does **not** use a Sigmoid function.

Instead, it outputs raw values called **logits**, which are processed internally by the CrossEntropy loss function.

---

# Training Pipeline

The training loop follows these steps for every mini-batch:

1. Forward pass
2. Compute CrossEntropy loss
3. Clear previous gradients
4. Backpropagation (`loss.backward()`)
5. Update model parameters (`optimizer.step()`)

PyTorch automatically computes every gradient using Autograd.

---

# Loss Function

The model uses:

```python
nn.CrossEntropyLoss()
```

This loss combines:

- Softmax
- Negative Log Likelihood

into a single numerically stable operation.

---

# Optimizer

Training is performed using Stochastic Gradient Descent (SGD).

```python
optim.SGD(model.parameters(), lr=0.01)
```

The optimizer updates every trainable parameter after each mini-batch.

---

# Performance

After training for 10 epochs:

- Training Accuracy ≈ **93%**
- Test Accuracy ≈ **94%**

The model generalizes well and does not show significant signs of overfitting.

---

# Evaluation

Model performance was evaluated using:

- Test Accuracy
- Individual predictions
- Softmax probabilities
- Confusion Matrix

The confusion matrix shows that most errors occur between visually similar digits

![Confusion Matrix](images/Confusion_Matrix.png)

This indicates that the model learned meaningful visual patterns rather than memorizing the dataset.

---

# Concepts Learned

This project introduced several fundamental Deep Learning concepts:

- PyTorch tensors
- Dataset
- DataLoader
- Mini-batches
- Sequential models
- Fully connected layers
- ReLU activation
- Logits
- CrossEntropy Loss
- SGD optimizer
- Automatic differentiation (Autograd)
- Backpropagation
- Training vs Evaluation mode
- Accuracy
- Confusion Matrix

---

# Technologies

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Scikit-learn

---

##### Ramiro Alvarez