# Symmetry Breaking & Weight Initialization

## The Symmetry Problem

If two neurons start with identical weights:

```
N4 = ReLU(0.5×N1 + 0.5×N2)
N5 = ReLU(0.5×N1 + 0.5×N2)
```

They receive the same gradient, update the same
way, and stay identical forever.

> Two neurons, one detector. Total waste.

This is called the **symmetry problem**.

---

## The Fix: Random Initialization

Weights start randomly different → neurons break
symmetry → each drifts toward detecting something
different during training.

This is the entire point of random init.
Not accuracy — just making sure neurons
aren't clones of each other from the start.

---

## But the Deeper Point Still Stands

*"If the task only needs one combination,
extra neurons are wasteful."*

Yes. And the field knows it. That's why these
techniques exist:

**Dropout**
Randomly kills neurons during training.
Forces the network not to rely on any one neuron,
preventing redundancy from forming.

**Regularization**
Pushes unnecessary weights toward zero,
effectively silencing redundant neurons.

**Pruning**
Literally removes redundant neurons after
training. Often 90%+ can be cut with
minimal accuracy loss.

---

## The Honest Picture

Fully connected layers are **wasteful by design**:

```
1. Throw too many neurons at the problem
2. Let training kill off the useless ones
3. Keep what matters
```

It's brute force, not elegance.

**CNNs, attention mechanisms, transformers** —
these exist precisely because researchers got tired
of this wastefulness and started building
structure directly into the architecture instead.

---

## Why Initialize Weights But Not Biases?

Weight initialization breaks symmetry by giving
each neuron a different starting point, so they
can each learn different patterns.

Biases don't cause the symmetry problem —
even if all biases are zero, neurons still
diverge because their weights differ.

So weight init is sufficient. Bias init
is not mandatory.

---

## Further Reading

[The Importance of Weight Initialization in MLP](https://ashutoshkriiest.medium.com/the-importance-of-weight-initialization-in-multi-layer-perceptron-mlp-for-artificial-intelligence-dbfc3d463b8e)