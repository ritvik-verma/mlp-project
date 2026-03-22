# He vs Xavier: Why Changing 1 to 2 Matters

## What Is Being Controlled?

Each layer computes:

```
Z = XW
```

The goal is to keep variance stable across layers:

```
Var(Z) ≈ Var(X)
```

So the signal neither:
- Shrinks layer by layer ❌ (vanishing)
- Blows up layer by layer ❌ (exploding)

---

## Xavier Init — `Var(W) = 1/n`

Designed for activations that keep values
symmetrically distributed. Works well for:

- `tanh`
- `sigmoid`

---

## He Init — `Var(W) = 2/n`

Designed for **ReLU**, which does:

```
ReLU(x) = max(0, x)
```

ReLU kills ~50% of values, so:

```
Var(after ReLU) ≈ ½ × Var(before)
```

He's fix: **double the variance at init** to compensate.

```
½ × (2/n) = 1/n  ← effective variance stays stable
```

---

## What Happens If You Don't Change 1 → 2?

**Xavier with ReLU:**
```
Var = 1/n
→ After ReLU: shrinks
→ After ReLU: shrinks again
→ After ReLU: shrinks again
→ vanishing activations → dead gradients → no learning
```

**He with ReLU:**
```
Var = 2/n
→ After ReLU: stays stable
→ After ReLU: stays stable
→ healthy signal flow throughout
```

---

## Why This Matters More in Deep Networks

In 2 layers → small effect.
In 10 layers → catastrophic effect.

Each ReLU halves the variance, so after 10 layers:

```
(½)^10 ≈ 0.001  →  almost zero signal
```

---

## Intuition

Think of each layer as:

```
signal × weight → ReLU → loses ~half its energy
```

- **Xavier** assumes no energy loss → signal dies slowly
- **He** compensates for the loss → signal survives

Initialization is fundamentally about
**energy preservation across layers.**

---

## TL;DR

| Init   | Formula  | Works with          |
|--------|----------|---------------------|
| Xavier | `1/n`    | tanh, sigmoid       |
| He     | `2/n`    | ReLU                |

ReLU kills ~50% of signal per layer.
He doubles the init variance to compensate.
Without it, activations shrink, gradients die,
and learning stalls — especially in deep nets.