# Part 4: Data Flow and Workflow

## 4.1 Data Format

The dataset used in this project originates from the RuNNer format, which stores atom-centered symmetry functions for each structure. Understanding the file layout is essential for modifying the parser or preparing new datasets.

### 4.1.1 File Structure

Each structure ("block") in `testing.data` consists of:

1. **Header line** — Contains the number of atoms in the block.
2. **Energy line** — Provides the reference total energy (in eV) and optional metadata.
3. **Atom lines** — One line per atom containing:
   - Element symbol (e.g., `La`, `O`, `Zr`)
   - Cartesian coordinates (optional in some datasets)
   - 159 symmetry function values (floating-point numbers)

Example snippet:

```
6
-0.25678123
La 0.001 0.002 0.003 0.195 ... 0.011
O  -0.012 0.077 -0.034 0.163 ... 0.025
...
```

The parser in `utils.read_file` reads the block header, energy, and iteratively parses each atom line. Symmetry function values are stored in NumPy arrays, later converted to PyTorch tensors within `BlockDS` for GPU processing.

### 4.1.2 Symmetry Function Representation

For each atom, the 159-dimensional vector includes both radial and angular descriptors. Indices are consistent across elements, meaning `SF-22` refers to the same parameterization regardless of species. This consistency enables direct comparison of feature importance across La, O, and Zr.

### 4.1.3 Energy Storage

Energies are stored in eV and normalized during preprocessing:

$$ E^{\text{norm}} = \frac{E - \mu}{\sigma} $$

where $\mu = -0.2628$ eV and $\sigma = 7.85$ meV for this dataset. Normalization stabilizes training, while predictions are later denormalized for reporting.

## 4.2 Training Workflow

Figure 2 provides a textual workflow diagram from raw data to trained model.

**Figure 2.** Training workflow (textual):

1. **Data Loading** → `utils.read_file` parses `testing.data` into block dictionaries.
2. **Normalization** → Energies normalized; symmetry vectors stored as tensors.
3. **Dataset Creation** → `BlockDS` wraps blocks and pre-loads tensors onto GPU.
4. **Data Splitting** → Deterministic 90/10 train-val split based on random seed.
5. **DataLoaders** → `DataLoader` objects provide mini-batches with custom `collate`.
6. **Model Initialization** → `AtomicMLP` constructed per config and moved to GPU.
7. **Training Loop** → For each epoch: run training forward/backward pass, evaluate validation set, log MAE.
8. **Checkpointing** → Save `exp_model.pth` and `exp_run_config.json` for reproducibility.
9. **Visualization** → Generate loss curves and parity plots for documentation.

Figure 2 mirrors the chronological events written to `SymFSel/exp_res/exp_run_config.json` when executing `python train.py --datafile ./data/testing.data --exp_prefix exp`, so each numbered stage corresponds directly to an artifact or log line captured during that run.

## 4.3 Feature Selection Workflow

Following model training, the feature selection stage uses the workflow summarized in Figure 3.

**Figure 3.** Feature selection workflow (textual):

1. **Load Run Directory** → Read configuration JSON, reconstruct network architecture, load weights.
2. **Load Feature Dataset** → Typically the same `testing.data`, though alternative validation/test sets can be supplied via `--datafile_fs`.
3. **Select Species** → Optional filtering (default analyzes all La, O, Zr).
4. **Compute Baseline Performance** → Evaluate MAE on the chosen dataset with all features.
5. **Apply Importance Method**:
   - *Weight-based*: Aggregate absolute weights for quick diagnostics.
   - *PFI*: For each feature, permute values within the species subset, recompute MAE, record degradation.
6. **Rank Features** → Sort features by importance score and save to text files.
7. **Top-k Evaluation (optional)** → For k from 1 to 159, retrain lightweight models or reuse predictions with mask to measure MAE degradation.
8. **Visualization** → Export bar plots and performance curves for each species.

This figure follows the exact invocation `python features.py --run_dir exp_res/ --datafile_fs ./data/testing.data --method pfi --eval_top_k` (see Terminal log and artifacts under `SymFSel/exp_res/`), ensuring that every arrow is backed by concrete CLI arguments and output files.

## 4.4 Data Provenance and Reproducibility

To ensure reproducibility, each run stores:

- `exp_run_config.json` — Contains data paths, model hyperparameters, normalization constants.
- `exp_model.pth` — PyTorch checkpoint with trained weights.
- Figures and text files — Provide human-readable summary of training dynamics and feature rankings.

By combining these artifacts with the original dataset, any researcher can reproduce the training, regenerate feature importance rankings, or extend the study to new descriptors.

---
