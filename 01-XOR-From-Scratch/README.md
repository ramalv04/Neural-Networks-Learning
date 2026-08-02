[Versión en Español](README_ES.md)

# Project 1 - Neural Network from Scratch (NumPy)

## Objective

Implement a fully from-scratch neural network using only NumPy.

The main goal of this project was not to achieve the best possible performance, but to deeply understand how a neural network works internally without using Deep Learning libraries such as PyTorch or TensorFlow.

All training logic was implemented manually.

---

# Problem

The network learns the XOR logical function.

| Input 1 | Input 2 | Output |
|--------:|--------:|-------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This problem is especially interesting because **it cannot be solved with a single neuron**, so it requires at least one hidden layer.

---

# Architecture

The network used was:

```
2 inputs
	 │
	 ▼
3 hidden neurons (Sigmoid)
	 │
	 ▼
1 output neuron (Sigmoid)
```

---

# Implemented Concepts

Everything was implemented manually:

- Forward propagation
- Sigmoid function
- Sigmoid derivative
- Mean Squared Error (MSE)
- Backpropagation
- Gradient Descent
- Weight and bias updates
- Random parameter initialization

No Machine Learning framework was used.

---

# Training Flow

Each epoch performs the following steps:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Gradient calculation
5. Weight and bias updates

```
Inputs
	│
	▼
Forward
	│
	▼
Prediction
	│
	▼
Loss
	│
	▼
Backpropagation
	│
	▼
Gradients
	│
	▼
Parameter updates
```

---

# Results

After training, the network correctly learns the XOR function.

Example output:

```
[[0.0077]
 [0.9839]
 [0.9839]
 [0.0207]]
```

The expected output was:

```
[[0]
 [1]
 [1]
 [0]]
```

The network successfully approximates the solution.

---

# Training Curve

During training, the loss decreases progressively until it converges.

![Loss curve image](images/loss.png)

---

# Learnings

This project helped me understand:

- What a neuron represents.
- What weights and biases are.
- How the forward pass is performed.
- How the loss function is calculated.
- How the backpropagation algorithm works.
- How gradient descent modifies parameters to minimize error.
- The influence of the learning rate.
- The importance of weight initialization.

---

# Technologies

- Python
- NumPy
- Matplotlib

---

##### Ramiro Alvarez

