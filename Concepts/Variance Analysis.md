## Variance Analysis & Initialization — Study Notes

> **Central question:** How does variance propagate through layers, and why does initialization make or break training?

---

### 🔴 What Went Wrong First — Uniform Initialization

**Method used:** `np.random.rand` (values in [0, 1])

**Why it failed:**
- All weights were positive → weighted sums became biased upward
- ReLU then collapsed most activations to zero
- Signal vanished before reaching later layers — no meaningful propagation

> **Root cause:** Bad initialization isn't just a *scale* problem. It's also a *symmetry* problem.

---

### ✅ What Fixed It — He Initialization

**Method:** `np.random.randn * sqrt(2/n)`

- Zero-centered normal distribution → symmetry restored
- Scale factor `sqrt(2/n)` compensates for ReLU's variance reduction
- Variance remained stable across all layers after switching

**Formula:**
```python
W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
```

---

### 📐 How ReLU Affects Variance

| Input Distribution | ReLU Behavior | Effect on Variance |
|---|---|---|
| Zero-centered (e.g., [-1, 1]) | Kills ~50% of activations | Reduces variance by ~½ |
| Non-centered (e.g., [0, 1]) | Passes most values through | Behaves almost linearly |

> **Insight:** Centering your data matters — but it also exposes you to dead neurons if initialization is off.

---

### 📊 Final Variance Results

| Layer Type | Variance Behavior |
|---|---|
| Linear layers | Preserved within same order of magnitude |
| Activation layers (ReLU) | Reduced, but did not collapse |

✅ Stable propagation confirmed end-to-end.

---

### 💡 Key Takeaways

1. **Initialization must address both scale AND distribution** — getting one right isn't enough
2. **He initialization** is the correct choice for ReLU-based networks
3. **Variance monitoring** across layers is a practical debugging tool
4. **Small details matter** — `rand` vs `randn` can completely break learning

---

### 🧠 The Big Picture

Signal flow through a neural network is fragile. Proper initialization isn't a fine-tuning step — it's a prerequisite for learning to happen at all. This experiment shows that understanding *why* an initialization works (not just *which* one to use) leads to better intuition for debugging and design.