# parametric-umap-pytorch
# Parametric UMAP from scratch (PyTorch)

A from-scratch implementation of **Parametric UMAP**: a neural encoder that learns to map 784-dimensional MNIST images to a 2D embedding while preserving the fuzzy neighbourhood graph of the original data. Built as a practical assignment in the Data Science School 2026 course.


## Result

| Metric | Value |
|---|---|
| Validation k-NN probe accuracy (best epoch) | **0.544** (epoch 34 of 40) |
| Validation k-NN probe accuracy (final epoch) | 0.523 |
| Training loss (epoch mean), start → end | 730.9 → 203.3 |
| Held-out split | 6,000 MNIST train images (never trained on) |
| Seed | 42 (deterministic) |

The probe is a 5-NN classifier fitted on a stratified half of the 2D embedding and scored on the other half. It measures whether the neighbourhoods of the 2D map still agree with the digit labels. Chance level for 10 classes is 0.10.

## What is implemented

All numerical core lives in `UMAP.py`, in pure PyTorch and differentiable end to end:

- **`compute_min_pairwise_distance`**: per-point nearest-neighbour distance (ρᵢ), diagonal excluded by masking with `inf`.
- **`find_a_b`**: fits the low-dimensional kernel `φ(d) = 1 / (1 + a·d^(2b))` to the UMAP target curve with SciPy (offline, once).
- **`find_sigma`**: vectorised binary search for each point's bandwidth σᵢ, so that the local fuzzy weights sum to log₂(k).
- **`compute_asymmetric_p`**: directed memberships `p_{j|i} = exp(-max(0, dᵢⱼ - ρᵢ) / σᵢ)`.
- **`compute_symmetric_p`**: fuzzy-set union `p + pᵀ - p∘pᵀ` (probabilistic t-conorm, not an average).
- **`umap_loss`**: fuzzy-set cross-entropy, with clamping for numerical stability, purely elementwise so it works on both dense blocks and flat edge lists, and passes `torch.autograd.gradcheck`.

Training `boilerplate.py` uses the pieces above as follows:

- **Global graph:** `generate_global_probabilities.py` builds an exact k-NN graph (k = 15) over all 54k training images and stores it as a sparse matrix, about 0.05% dense. The graph is built once on the full dataset. Building it per minibatch would make "neighbours" random strangers.
- **Edge sampling:** each step takes a batch of real graph edges (512 positives) and pairs every head with 5 negatives. Negatives come from a random cyclic shift of the batch, which keeps the backward pass deterministic on GPU.
- **Encoder:** small CNN, (B, 1, 28, 28) → (B, 2).
- **Optimiser:** Adam, lr 1e-3, 40 epochs, 100k sampled edges per epoch.
- **Model selection:** the checkpoint with the best validation probe score is kept, not the last epoch. The loss can keep falling while the embedding degrades, so the probe is the more reliable signal.

## Repository layout

```
UMAP.py                    # the implementation (UMAP maths)
boilerplate.py                      # data, CNN encoder, Lightning module, training loop
generate_global_probabilities.py    # builds the global sparse graph
selfcheck.py                        # structural tests (shapes, ranges, symmetry, gradients)
logs/metrics.csv                    # per-step training log from the run reported above
docs/training_curves.png            # plot of the log
```

## How to run

```bash
pip install torch lightning scikit-learn scipy torchvision numpy

python selfcheck.py                       # structural checks on student_tasks.py
python generate_global_probabilities.py   # build the global p_ij graph (once)
python boilerplate.py                     # train; logs val_knn_acc each epoch
```

Training runs on CPU, CUDA or MPS. The full run takes a few minutes on a GPU.

## What I learned

- **Why the graph has to be global.** A minibatch of 256 random images almost never contains real neighbours, so the attractive term of the loss has nothing to attract. Iterating over graph edges fixed the 600:1 imbalance between repulsive and attractive pairs.
- **Numerical details that break silently.** `log(0)` in the loss produces NaN that destroys every weight on the next optimiser step. Clamping inputs before the logs, and using masked `inf` instead of `0 * inf` on the diagonal, were the two fixes that mattered most.
- **Broadcasting.** Row-wise quantities (ρᵢ, σᵢ) must be broadcast with `unsqueeze(1)`. Getting this wrong transposes the whole graph without any error.
- **Metric choice.** Validation loss is not a good stopping criterion for an embedding. Selecting by held-out neighbourhood accuracy gave a better checkpoint than the last epoch.

