# Part 5: Implementation Details

## 5.1 GPU Optimization Strategies

### 5.1.1 Preloading Data onto GPU

The `BlockDS` dataset class immediately transfers symmetry vectors, species indices, and energies to the target device (CUDA when available). This design avoids repeated CPU→GPU copies during each batch, reducing the per-iteration overhead. A memory usage estimate is printed after loading to ensure the dataset fits within device constraints (~41 MB for this project).

### 5.1.2 Automatic Mixed Precision (AMP)

PyTorch's `torch.amp.GradScaler` enables mixed precision training. The implementation wraps forward and backward passes within `torch.autocast(device_type=dev, dtype=torch.float16)` contexts. Gradients are scaled before backpropagation and unscaled prior to optimizer steps, preventing underflow in FP16 arithmetic. On an RTX 4060 GPU, AMP reduces epoch time from ~0.11 s to ~0.07 s, yielding ~1.6× speedup without accuracy loss.

### 5.1.3 TF32 Acceleration

For Ampere GPUs, TensorFloat-32 (TF32) offers enhanced throughput while maintaining 10-bit mantissa precision. The script sets `torch.backends.cuda.matmul.allow_tf32 = True` to automatically leverage TF32 for matrix multiplications, further accelerating training.

### 5.1.4 Efficient Aggregation with `scatter_add_`

Element-specific networks produce per-atom energies that must be summed per structure. Instead of looping in Python, the model uses `torch_scatter.scatter_add` (or PyTorch's native `scatter_add_`) to accumulate contributions based on block identifiers. This vectorized aggregation enables efficient parallel reduction on the GPU.

## 5.2 Key Algorithms

### 5.2.1 Weighted MAE Calculation

To handle imbalanced energy distributions, each structure receives a weight computed from the histogram of normalized energies. The MAE loss for a batch becomes:

$$ \text{MAE}_{\text{weighted}} = \frac{\sum_{i} w_i |E_i^{\text{pred}} - E_i^{\text{ref}}|}{\sum_{i} w_i} $$

where weights $w_i$ are higher for rarely observed energy regions. The implementation applies weights element-wise and divides by the sum of weights within the batch, ensuring unbiased gradient magnitudes.

### 5.2.2 Permutation Feature Importance

Algorithm 1 outlines the PFI implementation used in `features.py`.

**Algorithm 1: Permutation Feature Importance**

1. Input: Trained model $M$, dataset $(X, y)$, feature index set $F$, species mask $S$, baseline MAE $E_{\text{base}}$.
2. For each feature $f \in F$:
   a. Copy dataset subset corresponding to species $S$.
   b. Permute the values in column $f$ across samples.
   c. Evaluate model: $E_f = \text{MAE}(M, X^{\pi_f}, y)$.
   d. Importance $I_f = E_f - E_{\text{base}}$.
3. Sort features by $I_f$ and return rankings.

The algorithm operates species-wise to respect element-specific networks. Importance scores are reported in meV to highlight subtle differences.

### 5.2.3 Top-k Performance Evaluation

To assess how many descriptors are necessary for accurate predictions, the script performs the following procedure for each species:

1. Sort features by PFI score (descending).
2. For $k = 1 \ldots 159$:
   a. Retain only the top $k$ features by zeroing out the remainder.
   b. Recompute MAE on the validation subset.
   c. Record degradation relative to the full-feature baseline.
3. Plot degradation vs. $k$ to visualize diminishing returns.

This approach identifies the elbow point around 50-70 features where additional descriptors provide minimal benefit.

## 5.3 Practical Considerations

- **Reproducibility**: Random seeds stabilize data splits and weight initialization. Both NumPy and PyTorch seeds are set via CLI argument `--seed`.
- **Checkpointing**: Model weights and configuration files are stored together so feature analysis can reconstruct the exact trained state.
- **Command-Line Automation**: Running `python train.py --exp_prefix exp` produces all artifacts in `exp_res/`, enabling consistent experiment naming.
- **Environment Management**: A dedicated `SymFSel` conda environment isolates dependencies (PyTorch, torch-scatter, matplotlib, NumPy, pandas) and ensures compatibility across machines.

---
