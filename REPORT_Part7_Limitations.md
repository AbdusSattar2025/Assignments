# Part 7: Limitations and Risk Assessment

## 7.1 Dataset Coverage

1. **Finite structural diversity**: The 1,460 Lanthanum Zirconate snapshots originate from closely related molecular dynamics windows. Compressive and tensile distortions near the perovskite phase transition are under-sampled, which explains the residual bias for high-energy La–O cages noted in Figure 9.
2. **No force supervision**: Energies alone constrain the potential surface weakly. Without force labels, the network may interpolate smoothly but could still yield inaccurate gradients when deployed in molecular dynamics.
3. **Species imbalance**: Oxygen atoms dominate the descriptor count, whereas each structure contains only four zirconium sites. The weighted MAE mitigates but does not fully eliminate the skew, so $R^2$ for Zr-centered environments may be overstated.

## 7.2 Modeling Assumptions

1. **Element-wise independence**: The Behler–Parrinello decomposition assumes additive atomic energies. Long-range electrostatic correlations (e.g., La–Zr coupling across multiple unit cells) are only indirectly captured via symmetry functions.
2. **Fixed descriptor set**: All 159 symmetry functions were inherited from earlier projects. Their parameters (cutoffs, $	heta$ grids) were not retuned to this dataset, so some information bottlenecks may stem from suboptimal descriptor placement rather than network capacity.
3. **Single architecture baseline**: Only the 256×2 AtomicMLP was explored. Without comparative baselines (e.g., message-passing networks), it is unclear whether the observed $R^2$ ceiling reflects data noise or architectural bias.

## 7.3 Feature-Selection Caveats

1. **Permutation leakage**: PFI permutes one feature at a time but keeps mixed-species descriptors synchronized. Subtle correlations between La and O channels might cause shared features (e.g., `SF-34`) to appear artificially strong.
2. **Top-$k$ masking heuristic**: Zeroing descriptors rather than retraining a reduced network exaggerates degradation for very small $k$. Table 3 should therefore be interpreted as a ranking aid, not a final prescription for descriptor pruning.
3. **Validation reuse**: The same 10% split serves both for early stopping and importance evaluation, introducing mild optimism. Cross-validation or bootstrapping would offer tighter confidence intervals.

## 7.4 Operational Constraints

1. **Hardware dependence**: AMP + TF32 optimizations target NVIDIA Ampere GPUs. Reproducing the exact timing or numerical behavior on AMD or older NVIDIA devices would require retuning precision strategies.
2. **Software stack stability**: The workflow relies on torch-scatter, which can lag behind mainline PyTorch releases. Future upgrades may break the current build unless the `SymFSel` environment is frozen.
3. **Config fragility**: Experiment metadata lives in JSON files under `exp_res/`. Manual edits or missing files can desynchronize the report from actual training runs unless checksums or schema validation are added.

## 7.5 Mitigation and Future Work

- Augment the dataset with high-temperature MD snapshots and include force labels to better constrain gradients.
- Re-optimize symmetry function hyperparameters using automated search (e.g., Bayesian optimization) before running PFI.
- Compare AtomicMLP against a lightweight equivariant GNN to test whether angular correlations are truly captured.
- Perform k-fold cross-validation for feature importance to obtain confidence intervals on the rankings.
- Package the training and analysis scripts as reproducible `invoke` tasks that rebuild environments and regenerate Figures 6–15 from scratch.

---
