# MLP from Scratch (NumPy) — Project Report

## Objective

To develop a deep, first-principles understanding of neural networks by implementing a Multi-Layer Perceptron (MLP) entirely from scratch using NumPy, including forward propagation, backpropagation, and training dynamics.

---

## Model Architecture

* **Input Layer:** 784 features (MNIST images)
* **Hidden Layers:**

  * Layer 1: 256 neurons
  * Layer 2: 128 neurons
* **Activation:** ReLU (with experimentation toward Leaky ReLU)
* **Output Layer:** 10 classes (Softmax)

---

## Core Implementations

### 1. Forward Propagation

* Implemented manually using matrix multiplication:

  * Linear transformations: `Z = XW + b`
  * Non-linearity via ReLU
  * Output probabilities via Softmax (numerically stabilized)

---

### 2. Loss Function

* Implemented Cross-Entropy Loss:

  * Per-sample loss:

    ```
    L = -sum(y * log(p))
    ```
  * Added epsilon for numerical stability
  * Reduced across classes (`axis=1`) and averaged across samples

---

### 3. Backpropagation

* Fully derived and implemented using chain rule:

  * `dZ3 = predictions - y`
  * Gradients computed layer-by-layer:

    * `dW = A^T @ dZ`
    * `db = sum(dZ)`
* Correct handling of:

  * ReLU derivative (element-wise masking)
  * Matrix dimensions and transposes
  * Jacobian interpretation (conceptual understanding)

---

### 4. Optimization

* Implemented Stochastic Gradient Descent (SGD):

  ```
  W = W - learning_rate * dW
  ```
* Added:

  * Learning rate scheduling
  * Gradient normalization (by batch size)

---

### 5. Training Pipeline

* Built full training loop:

  * Epoch-based training
  * Mini-batch gradient descent
  * Data shuffling per epoch
* Implemented:

  * Batch-wise forward + backward pass
  * Parameter updates per batch

---

## Debugging & Stability Handling

### Key Issues Encountered & Resolved

* Softmax numerical instability → fixed with max-shift
* Log(0) issue in cross-entropy → added epsilon
* Gradient explosion (NaNs) → fixed via learning rate tuning
* Incorrect batching (no shuffle) → identified and fixed

---

## Experimental Observations

### 1. Training Dynamics

* Full-batch GD → smooth but slower convergence
* Mini-batch GD → noisier but better generalization

---

### 2. Learning Rate Effects

* High learning rate → faster learning but unstable
* Mini-batch required lower learning rate for stability

---

### 3. Data Ordering Insight

* Non-shuffled data led to class-wise learning (implicit curriculum learning)
* Proper shuffling removed bias and improved generalization

---

## Performance

* Full-batch training: ~89–90% accuracy
* Mini-batch training (corrected): ~92–95% expected
* Observed peak (~97%) under structured (non-random) batching conditions

---

## Key Learnings

* Deep understanding of:

  * Gradient flow across layers
  * Role of activation functions
  * Matrix operations in neural networks

* Practical insights into:

  * Numerical stability (softmax, log)
  * Optimization behavior (SGD, batch size)
  * Training dynamics (data ordering, randomness)

---

## Conclusion

This project successfully demonstrates a fully transparent neural network system built from scratch. Beyond implementation, it provides strong intuition for how neural networks learn, behave, and stabilize.

It marks the transition from theoretical understanding to practical control over training — forming a strong foundation for advanced topics such as adaptive optimizers, regularization, and convolutional networks.

---
