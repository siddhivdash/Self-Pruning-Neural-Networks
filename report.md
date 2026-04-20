# Self-Pruning Neural Network — Case Study Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The gate for each weight is defined as:

    gate = sigmoid(gate_score)   ∈ (0, 1)

The sparsity loss adds a penalty proportional to the **sum of all gate values** (L1 norm):

    SparsityLoss = Σ gate_i  =  Σ sigmoid(gate_score_i)

During backpropagation, the gradient of this loss w.r.t. each gate_score is:

    ∂SparsityLoss / ∂gate_score_i  =  sigmoid'(gate_score_i)  =  gate_i × (1 − gate_i)

This gradient is always **positive**, meaning it always pushes gate_scores **downward**.
As gate_score → −∞, sigmoid(gate_score) → 0, and the weight is effectively removed.

**Why L1 and not L2?**  
An L2 penalty (sum of gate²) would produce a gradient of `2 × gate_i × (1 − gate_i)`,
which shrinks toward zero as the gate approaches zero — slowing down pruning right when
you need it most. The L1 penalty maintains a constant push direction regardless of
magnitude, which drives gates to **exactly zero** rather than just small values.
This is the key mathematical reason L1 is the standard choice for inducing sparsity.

---

## Results Table

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:-----------:|:-----------------:|:------------------:|
| 0.01        | 56.59             | 58.25              |
| 0.10        | 56.22             | 92.00              |
| 0.50        | 54.64             | 97.72              |

### Analysis of the λ Trade-off

**λ = 0.01 (Low regularization)**  
The sparsity penalty is weak relative to the classification loss.  
The network prunes ~58% of connections while maintaining the best accuracy (56.59%).  
Gates settle into a bimodal distribution with a meaningful surviving cluster.

**λ = 0.1 (Medium regularization)**  
A tenfold increase in λ drives sparsity to 92% with only a 0.37% accuracy drop.  
This demonstrates the efficiency of the gated pruning mechanism — the network
identifies and discards ~3.5M of 3.8M connections without significant performance loss.

**λ = 0.5 (High regularization)**  
97.72% of gates are driven to near-zero. The network is operating on fewer than
90,000 active connections out of 3.8 million. Accuracy drops only ~2% compared
to λ=0.01, showing the network's most critical connections are preserved.

**Key insight:** The accuracy is remarkably stable (54–57%) across a 50× range of λ,
while sparsity varies from 58% to 98%. This suggests the network has significant
redundancy — most connections contribute little to classification performance.

---

## Gate Distribution

The plot for the best model (λ=0.01) shows:
- A **dominant spike at gate ≈ 0**: the majority of weights have been pruned
- A **small surviving cluster near gate ≈ 1**: the network's critical connections
- A near-empty middle region: gates decisively collapsed to one extreme or the other

This bimodal "all-or-nothing" behavior confirms the L1 penalty is working as intended —
it does not merely shrink weights uniformly, but creates a clear binary decision boundary
between pruned and preserved connections.

---

## Training Procedure

**Stage 1 — Warm-up (λ=0, 8 epochs):**  
The network trains normally with no sparsity pressure. This allows weights and gates
to reach a meaningful starting point before pruning begins, preventing the network
from collapsing into a degenerate sparse state before it has learned anything useful.

**Stage 2 — Pruning (λ>0, 12 epochs):**  
The sparsity loss is activated. Gate-specific optimizer learning rate is set 100×
higher than weight learning rate (0.1 vs 0.001), ensuring the sparsity gradient
can compete with the classification gradient flowing through the same parameters.

**Total Loss = CrossEntropyLoss + λ × SparsityLoss**