# Weight Initialization: Why Random Init Fails

## The Problem with Naive Random Init

```python
W = (np.random.rand(...) - 0.5) * 0.01
```

This gives a uniform distribution in **[-0.005, 0.005]**.

---

## Problem 1: Scale Is Way Too Small

Estimated variance of this init:

```
Var ≈ (0.005)² ≈ 2.5e-5
```

Compare that to **He initialization** (for the first layer):

```
Var ≈ 2 / 784 ≈ 0.0025
```

> That's **~100× larger** than the naive init.

### What happens because of this

- **Layer 1:** `Z1 = XW` → very small values
- **ReLU:** many activations ≈ 0
- **Layer 2:** values shrink even further
- **Result:** signal dies quickly

This is the classic **vanishing signal problem** —
activations near zero, gradients near zero, learning
extremely slow or completely stuck.

---

## Problem 2: Not Scaled to Input Size

Naive init uses the same `0.01` for all layers.

The correct logic: **scale depends on `n_in`**.

Why? Because:

```
Z = w₁x₁ + w₂x₂ + ... + wₙxₙ
```

More inputs → more accumulation → larger variance
is needed to keep the signal balanced.

---

## Problem 3: Ignores the Activation Function

The init assumes nothing about the activation,
but you're using **ReLU** — which kills half the signal
on every forward pass.

**He init** specifically accounts for this by scaling
weights so that the signal survives through ReLU.

---

## Deep Intuition: Why This Matters

With weights too small, the network starts nearly
linear and near-zero:

```
f(x) ≈ constant / tiny variation
```

So early in training:

- Forward pass produces near-zero activations
- Gradients carry almost no information
- Learning effectively stalls

### The Whisper Analogy

```
Your init   →  whispering signal
ReLU        →  cuts half of it
Next layer  →  whispers even quieter
Layer 3+    →  silence
```

---

## What He Init Fixes

He initialization scales weights so that:

```
Var(output) ≈ Var(input)
```

...even **after** ReLU. The signal stays alive
across all layers.

---

## Summary

| Property            | Naive Init      | He Init          |
|---------------------|-----------------|------------------|
| Variance            | ~2.5e-5 (tiny)  | ~0.0025 (100×)   |
| Scaled to `n_in`?   | ❌ No           | ✅ Yes            |
| Accounts for ReLU?  | ❌ No           | ✅ Yes            |
| Breaks symmetry?    | ✅ Yes          | ✅ Yes            |
| Preserves signal?   | ❌ No           | ✅ Yes            |

> **Key takeaway:** Symmetry breaking alone is not enough.
> Signal preservation is equally critical.