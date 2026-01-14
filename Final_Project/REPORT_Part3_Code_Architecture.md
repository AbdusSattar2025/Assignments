# Part 3: Code Architecture Analysis

## 3.1 Module Overview

The project codebase resides inside the `SymFSel` directory and follows a modular structure that separates configuration, model definitions, utility routines, training scripts, and feature analysis tools. Figure 1 summarizes the high-level architecture of the system.

**Figure 1.** High-level module interaction diagram (conceptual).  
`config.py` → provides global settings used by `train.py` and `features.py`.  
`models.py` → defines neural network models for training and feature analysis.  
`utils.py` → supplies common routines (data loading, weighting, plotting) reused across scripts.  
`train.py` → orchestrates dataset preparation, model training, loss logging, checkpointing, and scatter plot generation.  
`features.py` → loads the trained checkpoint and performs feature importance evaluation.

Figure 1 is reconstructed from the actual dependency graph observed while running `python train.py --exp_prefix exp` and the subsequent `features.py` analysis in `SymFSel`. Each arrow therefore denotes a real file-level interaction: configuration constants originating in `config.py` feed both execution scripts, while `models.py` and `utils.py` supply the concrete neural network instances and helper routines referenced by the logs stored in `SymFSel/exp_res/exp_run_config.json`.

## 3.2 Detailed Component Analysis

### A. Configuration Module (`config.py`)

The configuration module centralizes file paths, default hyperparameters, and global constants. Table 1 lists the most relevant parameters.

**Table 1.** Key settings defined in `config.py`.

| Category | Parameter | Value / Description |
| --- | --- | --- |
| Paths | `DATA_DIR` | `./data` (base folder for datasets) |
|  | `DEFAULT_DATAFILE` | `data/blocos_real.dat` (overridden via CLI) |
| Energy | `ENERGY_IDX` | Column index containing total energy |
| Model | `DEFAULT_MODEL_TYPE` | `mlp` (atomic neural networks) |
|  | `DEFAULT_HIDDEN_UNITS` | 256 neurons per hidden layer |
|  | `DEFAULT_DEPTH` | 2 hidden layers |
| Training | `DEFAULT_LEARNING_RATE` | 3e-4 |
|  | `DEFAULT_EPOCHS` | 500 (CLI sets 300) |
|  | `DEFAULT_BATCH_SIZE` | 1024 symmetry-function vectors |
|  | `DEFAULT_VALIDATION_RATIO` | 0.2 (CLI uses 0.1) |
| Feature weighting | `WEIGHTING_N_BINS` | 40 histogram bins |
|  | `WEIGHTING_POW` | 1.0 (controls inverse-frequency emphasis) |
| Plotting | `DEFAULT_PLOT_SET` | `both` (training + validation) |
| Hardware | `PREFERRED_DEVICE` | `cuda` if available |

Each constant is imported by both `train.py` and `features.py`, ensuring consistent defaults when command-line arguments are omitted. The periodic table array, `PTABLE`, provides element symbols used when parsing datasets and reporting species-specific results.

### B. Model Architectures (`models.py`)

The `models.py` file defines three neural network classes tailored to atomic energy prediction:

1. **`SpeciesMLP`** — encapsulates a fully connected multilayer perceptron dedicated to a single chemical species. The constructor accepts the input dimension (`n_g` = 159 symmetry functions), hidden width, depth, and activation choice (GELU). Forward propagation processes the input through stacked linear layers and outputs a scalar atomic energy.

2. **`AtomicMLP`** — extends `SpeciesMLP` to multi-element systems by instantiating one sub-network per element. During the forward pass, the dataset provides per-atom symmetry vectors and species indices. The module routes each atom to the correct sub-network, then aggregates predicted energies using `torch_scatter.scatter_add`, ensuring differentiability and efficient GPU execution. This design reflects the Behler-Parrinello philosophy of element-specific neural networks.

3. **`AtomicLin`** — implements a lightweight linear baseline consisting of a single linear layer without hidden units. Although less expressive, it offers diagnostic value when verifying training pipelines or benchmarking the benefit of nonlinear architectures.

The models share a common interface that accepts concatenated symmetry vectors `(N_atoms × n_g)`, species indices, and block identifiers for aggregation. This uniformity simplifies integration with the training loop.

### C. Utility Functions (`utils.py`)

The utilities module delivers reusable functionality critical to both training and feature analysis:

- **Data Loading (`read_file`)**: Parses the custom `.data` file format that lists atomic blocks. Each block begins with metadata lines followed by atom entries containing element symbols and 159 symmetry function values. The parser normalizes line endings, converts strings to numeric tensors, and returns a list of blocks plus the per-structure energy array.

- **Sample Weighting (`make_weights`)**: Computes inverse-frequency weights for each structure based on histogram binning of normalized energies. This technique mitigates the imbalance between low-energy ground states and higher-energy configurations, ensuring the MAE loss treats all regions of energy space fairly.

- **Training Loop Helper (`run_epoch`)**: Consolidates the logic for iterating over a dataloader, performing forward/backward passes, updating gradients with AMP, and aggregating MAE metrics. The function returns MAE, predictions, and reference values, enabling both training and validation workflows to reuse identical code paths.

- **Visualization (`plot_loss_curve`, `scatter`)**: Generates publication-quality figures for loss evolution and parity plots. The scatter utility computes R² on the fly, adds a dashed y=x reference line, and saves PNGs referenced later in the report.

### D. Training Pipeline (`train.py`)

The training script brings all modules together and exposes a command-line interface. Major stages include:

1. **Argument Parsing**: Users specify data paths, model depth, hidden units, training epochs, batch size, validation ratio, random seed, AMP usage, weighting, and output prefix. This flexibility allowed the current study to set `--val_ratio 0.1`, `--epochs 300`, and `--exp_prefix exp`.

2. **Data Preprocessing**: The script loads the dataset via `utils.read_file`, normalizes energies (subtract mean, divide by standard deviation), constructs a `BlockDS` dataset class, and splits data into train/validation subsets with deterministic shuffling.

3. **Custom Dataset & DataLoader**: `BlockDS` converts raw block dictionaries into GPU tensors. The `collate` function concatenates atom tensors, builds block identifiers, and returns batched data ready for the model. DataLoaders use `prefetch_factor` for performance when workers > 0.

4. **Training Loop**: For each epoch, the script runs training and validation phases using `utils.run_epoch`, converts MAE back to physical units, stores history for plotting, and logs epoch summaries. Automatic Mixed Precision (AMP) and AdamW provide efficient optimization.

5. **Checkpointing & Configuration Saving**: After training, the script writes `exp_model.pth` alongside `exp_run_config.json`, capturing all hyperparameters, dataset paths, and normalization statistics for reproducibility.

6. **Result Visualization**: Loss curves and scatter plots are saved automatically in the `exp_res` folder. These figures serve as the primary artifacts referenced in the Results section.

### E. Feature Analysis (`features.py`)

The feature analysis script operates post-training to quantify descriptor importance:

1. **Run Directory Loading**: Reads `exp_run_config.json`, reconstructs the AtomicMLP architecture with identical hyperparameters, and loads the trained weights.

2. **Dataset Preparation**: Accepts a data file (typically validation set) and optional species list. Symmetry vectors can be standardized before importance calculation when using weight-based methods.

3. **Importance Methods**:
   - *Weight-based*: Computes aggregated absolute weights from the first linear layer to approximate each feature's influence.
   - *Permutation Feature Importance (PFI)*: For each feature and species, randomly permutes the feature values and measures the resulting MAE increase. This approach, implemented in `evaluate_pfi`, provides reliable rankings for nonlinear models.

4. **Top-k Evaluation**: When `--eval_top_k` is enabled, the script progressively retains only the top K features and re-evaluates MAE. The resulting degradation curves quantify how many descriptors are necessary for near-optimal performance.

5. **Outputs**: The script saves bar charts (`features_<element>_pfi_importance.png`), text files containing ranked features, and performance-degradation plots (`performance_vs_features_<element>.png`). These artifacts feed directly into Section 6 of the report.

---
