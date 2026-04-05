## 🧠 Measuring Variance in Neural Networks — What Are You Actually Measuring?

> **Core confusion:** `np.var(layer, axis=1)` does NOT tell you what a layer did to your data. Here's why that matters.

---

### ❌ What You *Think* You're Measuring

**Goal:** "What did Layer 1 do to the data?"

**Code used:**
```python
np.var(layer, axis=1)  # axis=1 → across neurons, per sample
```

**What it actually computes:**
For each sample, how spread out are the neuron activations **within that one sample**.

> In other words: at a single moment in time (one input), are the neurons firing at similar levels, or are some much louder than others?

---

### ✅ What You *Should* Be Measuring

To understand what a layer did to the data, ask:

> **"How much does each neuron vary across different inputs?"**

**Code:**
```python
np.var(layer, axis=0)  # axis=0 → across samples, per neuron
```

**What this computes:**
For each neuron, how much does its activation change as different inputs pass through?

- **High variance** → this neuron is sensitive and informative
- **Low variance** → this neuron fires roughly the same for everything (possibly dead or redundant)

---

### 🔁 The Axis Trap — Quick Reference

| Code | Axis | Measures | Useful for |
|------|------|----------|------------|
| `np.var(layer, axis=1)` | across neurons | spread *within* one sample | checking activation spread per input |
| `np.var(layer, axis=0)` | across samples | spread *across* inputs | diagnosing dead neurons, layer health |

---

### 💡 Key Insight

**`axis=1`** answers: *"Are this sample's neurons spread out?"*  
**`axis=0`** answers: *"Is this neuron doing useful work across the dataset?"*

When debugging layers, you almost always want `axis=0`.