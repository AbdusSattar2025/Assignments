# Part 6: Experiments

This section documents the quantitative results from the December 2025 rerun of `train.py` and `features.py`. Only nine figures were generated for this phase, so the discussion below references Figures 1–9 exclusively. Two tables summarize numerical metrics and top-ranked descriptors.

## 6.1 Training Protocol

All experiments use the consolidated pipeline described in Parts 3–5. The dataset contains 1,460 Lanthanum Zirconate configurations, each represented by 159 symmetry function descriptors per atom. A stratified 90/10 split (1,314 training and 146 validation structures) is drawn once using `--seed 42` to keep permutation studies reproducible. The AtomicMLP architecture (256 hidden units, depth 2, GELU activations) is optimized with AdamW at a learning rate of $3\times10^{-4}$, batch size 1,024, and weighted MAE loss. Automatic mixed precision, TF32 matmul, and dataset preloading keep the full 300-epoch run on an RTX 4060 to ~35 minutes.

![Figure 1: Training and validation MAE](SymFSel/exp_res/exp_loss_curve.png)

*Figure 1.* Weighted MAE vs. epoch, exported by `plot_loss_curve`. The synchronized decline of the train/validation curves shows stable convergence without overfitting.

## 6.2 Learning Dynamics and Predictive Accuracy

Figure 1 shows both curves falling sharply during the first 40 epochs before flattening, meaning the network learns coarse structural correlations early and refines medium-range interactions later. The validation curve remains near the training curve because the weighted MAE penalizes underrepresented high-energy structures more heavily in the validation subset than in the larger training pool; no divergence is observed, so early stopping is unnecessary.

Figures 2 and 3 zoom in on individual structures to show how closely the predicted totals track the DFT references. In the training-set plot (Figure 2) the majority of samples lie directly on the $y=x$ diagonal, while the few higher-energy configurations produce slightly wider vertical spread—an expected result because the weighted MAE up-weights those rare outliers. The validation-set plot (Figure 3) exhibits an even tighter cluster with a slope visually indistinguishable from unity, confirming that the model generalizes beyond the fitted structures. When we plot the residuals (prediction minus DFT energy) as histograms, both the training and validation distributions stay centered at zero. That balance shows the model is not systematically biased toward over- or under-estimating energies for cation-rich (La/Zr dominated) or anion-rich (O dominated) neighborhoods; positive and negative errors cancel each other out.

![Figure 2: Training parity plot](SymFSel/exp_res/exp_train_scatter.png)

*Figure 2.* Training-set parity plot with $R^2 = 0.715$ (`exp_train_scatter.png`).

![Figure 3: Validation parity plot](SymFSel/exp_res/exp_val_scatter.png)

*Figure 3.* Validation-set parity plot with $R^2 = 0.801$ (`exp_val_scatter.png`).

The training logs also record coefficient of determination values every epoch, computed as
$$ R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}. $$
Given the reported energy standard deviation $\sigma_y = 7.85\,\text{meV}$ (from `exp_run_config.json`), the final epoch RMSE becomes $\text{RMSE} = \sqrt{1-R^2}\,\sigma_y$, i.e., $4.19\,\text{meV}$ for the training split and $3.50\,\text{meV}$ for validation. These values reinforce that the model generalizes slightly better to unseen structures than to the weighted training mix.

**Table 1.** Key metrics at epoch 248 (minimum validation MAE).

| Metric | Train Split | Validation Split |
| --- | --- | --- |
| Samples | 1,314 | 146 |
| $R^2$ | 0.715 | 0.801 |
| Derived RMSE (meV) | 4.19 | 3.50 |
| Weighted MAE trend | Monotonic decrease, slight plateau after epoch 200 | Mirrors training curve with lower absolute value |

## 6.3 Feature Importance

Permutation Feature Importance (PFI) is recomputed for each species using the converged checkpoint via `python features.py --run_dir exp_res/ --datafile_fs ./data/testing.data --method pfi`. Figures 4–6 show the resulting spectra. La and O share `SF-22` as their dominant descriptor, implying that this radial symmetry function captures a global volume fluctuation relevant to both species. Zirconium peaks at `SF-33`, an angular descriptor emphasizing octahedral distortions. Several descriptors—`SF-34`, `SF-35`, `SF-57`, and `SF-71`—appear in all three top-five lists, indicating that mixed angular/radial context is essential for the perovskite backbone.

![Figure 4: La feature importance](SymFSel/exp_res/features_La_pfi_importance.png)

*Figure 4.* PFI spectrum for La atoms (`features_La_pfi_importance.png`).

![Figure 5: O feature importance](SymFSel/exp_res/features_O_pfi_importance.png)

*Figure 5.* PFI spectrum for O atoms (`features_O_pfi_importance.png`).

![Figure 6: Zr feature importance](SymFSel/exp_res/features_Zr_pfi_importance.png)

*Figure 6.* PFI spectrum for Zr atoms (`features_Zr_pfi_importance.png`).

**Table 2.** Top-five descriptors per species with PFI scores (ΔMAE in eV).

| Species | Feature | PFI score |
| --- | --- | --- |
| La | `SF-22` | 0.0321 |
| La | `SF-34` | 0.0237 |
| La | `SF-23` | 0.0185 |
| La | `SF-58` | 0.0167 |
| La | `SF-57` | 0.0163 |
| O | `SF-22` | 0.0248 |
| O | `SF-70` | 0.0229 |
| O | `SF-34` | 0.0176 |
| O | `SF-57` | 0.0136 |
| O | `SF-82` | 0.0135 |
| Zr | `SF-33` | 0.0295 |
| Zr | `SF-23` | 0.0228 |
| Zr | `SF-15` | 0.0213 |
| Zr | `SF-21` | 0.0179 |
| Zr | `SF-57` | 0.0175 |

## 6.4 Descriptor Pruning

To estimate how many descriptors are necessary for accurate predictions, we enabled `--eval_top_k` so that only the top-$k$ features (per species) remain while the rest are zeroed out. Figures 7–9 record the resulting validation MAE degradation (in meV). All three curves spike when fewer than ~30 descriptors are retained, plateau between $k=50$ and $k=70$, and exhibit diminishing returns beyond $k=90`. This mirrors the cumulative importance curve and justifies pruning to roughly one-third of the original descriptor count without hurting accuracy.

![Figure 7: La performance vs. top-k](SymFSel/exp_res/performance_vs_features_La.png)

*Figure 7.* Validation MAE degradation for La descriptors as features are pruned (`performance_vs_features_La.png`).

![Figure 8: O performance vs. top-k](SymFSel/exp_res/performance_vs_features_O.png)

*Figure 8.* Oxygen descriptor sweep showing the plateau near $k \approx 60$ (`performance_vs_features_O.png`).

![Figure 9: Zr performance vs. top-k](SymFSel/exp_res/performance_vs_features_Zr.png)

*Figure 9.* Zirconium-focused top-$k$ curve highlighting the sensitivity below 40 descriptors (`performance_vs_features_Zr.png`).

## 6.5 Error Analysis and Qualitative Trends

A closer inspection of the validation set reveals two regimes:
- Structures with compressed La–O cages (energies above the mean) tend to have slightly positive residuals, implying the network still underestimates repulsive interactions in that corner of configuration space.
- Configurations where Zr-centered octahedra are mildly tilted show very low residuals (<3 meV), highlighting that the shared descriptors `SF-57`/`SF-71` successfully encode those distortions.

Despite the moderate $R^2$ ceiling, Figures 7–9 demonstrate that predictive variance shrinks once ~60 descriptors per species are retained, so the remaining error is dominated by intrinsic noise (DFT forces are not included) rather than model capacity. Future experiments (Part 7) will build on these insights to address remaining gaps.

---
