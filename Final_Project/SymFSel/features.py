# features.py (versão atualizada)

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

import config as project_config
import models
import utils

# --- Helper Functions ---

def plot_feature_importance(importances, feature_names, species_name, method_name, run_dir):
    """Plots and saves the feature importance bar chart."""
    if not feature_names: feature_names = [f"Feature {i+1}" for i in range(len(importances))]
    if isinstance(importances, torch.Tensor): importances = importances.cpu().numpy()
    importances_np = np.asarray(importances)
    if importances_np.ndim == 0: # Handle scalar case
        importances_np = np.array([importances_np.item()])
    if len(importances_np) == 0:
        print(f"Warning: No importances to plot for {species_name} using {method_name}.")
        return
    
    sorted_indices = np.argsort(importances_np)[::-1]
    
    plt.figure(figsize=(10, max(6, len(importances_np) * 0.3)))
    display_feature_names = np.array(feature_names)[sorted_indices] if len(feature_names) == len(importances_np) else [f"F{i}" for i in sorted_indices]

    base_name = f"features_{species_name}_{method_name}_importance"
    out_txt = os.path.join(run_dir, base_name + ".txt")
    out_png = os.path.join(run_dir, base_name + ".png")

    plt.barh(display_feature_names, importances_np[sorted_indices], color='skyblue')
    plt.xlabel(f"Importance ({method_name})")
    plt.ylabel("Symmetry Function Index")
    plt.title(f"Feature Importance for Species '{species_name}' ({method_name})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved feature importance plot to {out_png}")

    with open(out_txt, "w") as f:
        f.write("Feature\tImportance\n")
        for idx in sorted_indices:
            name = feature_names[idx] if idx < len(feature_names) else f"F{idx}"
            imp = importances_np[idx]
            f.write(f"{name}\t{imp:.6f}\n")
    print(f"Saved feature importances to {out_txt}")


def get_g_vectors_for_species(blocks_data, target_species_symbol, ptable_for_z2sym, standardize=False, g_mean=None, g_std=None):
    """Extracts all g-vectors for a specific atomic species from the data."""
    g_vectors = []
    for block in blocks_data:
        for atom_sym_from_data, atom_g in block["atoms"]:
            if atom_sym_from_data == target_species_symbol:
                g_vectors.append(atom_g)
    if not g_vectors:
        print(f"Warning: No g-vectors found for species {target_species_symbol}.")
        return np.array([])
    g_vectors_np = np.array(g_vectors, dtype=np.float32)
    if standardize:
        if g_mean is None or g_std is None:
            raise ValueError("g_mean and g_std must be provided for standardization.")
        g_vectors_np = (g_vectors_np - g_mean) / (g_std + 1e-8)
    return g_vectors_np


def calculate_g_stats(blocks_data):
    """Calculates the mean and standard deviation of all g-vectors in the dataset."""
    all_g_vectors = []
    for block in blocks_data:
        for _, atom_g in block["atoms"]:
            all_g_vectors.append(atom_g)
    if not all_g_vectors:
        return None, None
    all_g_vectors_np = np.array(all_g_vectors, dtype=np.float32)
    return np.mean(all_g_vectors_np, axis=0), np.std(all_g_vectors_np, axis=0)


def evaluate_performance_by_feature_count(sub_model, g_tensor, sorted_feature_indices, device, energy_std=1.0):
    """
    Evaluates sub-model performance degradation as features are added one by one based on importance.
    The result is returned in milli-electron-volts (meV).
    """
    sub_model.eval()
    n_features = g_tensor.shape[1]
    performance_results = []

    with torch.no_grad():
        # 1. Calculate the baseline prediction using ALL features
        baseline_preds = sub_model(g_tensor)

        # 2. Iteratively evaluate with top-k features
        for k in tqdm(range(1, n_features + 1), desc=f"Evaluating top-k features for {sub_model.__class__.__name__}", leave=False):
            # Get the indices of the top k features
            top_k_indices = sorted_feature_indices[:k]

            # Create a mask to zero out non-top-k features
            mask = torch.zeros(n_features, device=device)
            mask[top_k_indices] = 1.0

            # Apply the mask to the input features
            masked_g_tensor = g_tensor * mask
            
            # Get predictions with the masked features
            k_feature_preds = sub_model(masked_g_tensor)

            # MUDANÇA AQUI: Adicionamos '* 1000' para converter de eV para meV
            mae_degradation_in_meV = torch.abs(k_feature_preds - baseline_preds).mean().item() * energy_std * 1000
            performance_results.append((k, mae_degradation_in_meV))

    return performance_results


def plot_performance_vs_feature_count(k_values, error_values, species_name, run_dir, energy_std=1.0):
    """Plots the performance degradation curve, with y-axis in energy units (meV)."""
    plt.figure(figsize=(10, 6))
    
    # MUDANÇA AQUI: Alteramos o rótulo da unidade de 'eV' para 'meV'
    unit_label = "meV" if energy_std != 1.0 else "Normalized Units"
    plt.ylabel(f"Performance Degradation (MAE in {unit_label})")
    
    plt.plot(k_values, error_values, marker='.', linestyle='-', color='b')
    plt.xlabel("Number of Top Features Used")
    plt.title(f"Model Performance vs. Number of Features for Species '{species_name}'")
    plt.grid(True, which='both', linestyle=':')
    
    # MUDANÇA AQUI: A legenda também reflete a nova unidade
    min_degradation_val = min(error_values)
    plt.axhline(y=min_degradation_val, color='g', linestyle='--', 
                label=f'Min Degradation: {min_degradation_val:.4f} {unit_label}')
    plt.legend()
    
    plt.tight_layout()
    
    base_name = f"performance_vs_features_{species_name}.png"
    out_png = os.path.join(run_dir, base_name)
    plt.savefig(out_png)
    plt.close()
    print(f"Saved performance vs. feature count plot to {out_png}")


def main_feature_selector():
    parser = argparse.ArgumentParser(description="Feature importance evaluator. Loads config from a run directory.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the training run directory (containing model.pth and run_config.json).")
    parser.add_argument("--datafile_fs", type=str, required=True,
                        help="Path to the data file for feature selection analysis (e.g., validation or test set).")
    parser.add_argument("--species", type=str, nargs='*', default=None,
                        help="Optional. List of species symbols to analyze (e.g., O La Zr). "
                             "If not provided, all species from the trained model will be analyzed.")
    parser.add_argument("--method", choices=["weights", "pfi"], default="pfi",
                        help="Feature importance method: 'weights' or 'pfi'.")
    parser.add_argument("--standardize_g", action="store_true",
                        help="Standardize g-vectors (from --datafile_fs) before analysis. Recommended for 'weights' method.")
    parser.add_argument("--eval_top_k", action="store_true",
                        help="Evaluate and plot model performance vs. number of top-k features.")

    args = parser.parse_args()

    run_directory = pathlib.Path(args.run_dir)
    if not run_directory.is_dir():
        print(f"Error: Run directory not found: {run_directory}"); return

    possible_config_files = list(run_directory.glob("*run_config.json"))
    if not possible_config_files:
        print(f"Error: No '*run_config.json' file found in {run_directory}"); return
    config_file_path = possible_config_files[0]
    
    print(f"Loading run configuration from: {config_file_path}")
    with open(config_file_path, 'r') as f: run_config_loaded = json.load(f)

    # Load parameters from config
    model_type = run_config_loaded.get("model_type")
    n_g_from_config = run_config_loaded.get("n_g")
    species_list_from_training = run_config_loaded.get("species_list_ordered")
    s2i_from_training = run_config_loaded.get("s2i_map_used")
    num_species_for_model = len(species_list_from_training)
    hidden_units = run_config_loaded.get("hidden_units")
    depth_mlp = run_config_loaded.get("depth")
    
    # Load energy_std for scaling results back to eV
    energy_std_from_config = run_config_loaded.get("energy_std")
    if energy_std_from_config is None:
        print("Warning: 'energy_std' not found in config. Degradation will be shown in normalized units.")
        energy_std_for_scaling = 1.0 # Use 1.0 so it has no effect on multiplication
    else:
        energy_std_for_scaling = float(energy_std_from_config)
        print(f"Using energy_std = {energy_std_for_scaling:.4f} to scale degradation results to original energy units.")

    model_filename_from_config = run_config_loaded.get("model_path_relative", "model.pth")
    model_path = run_directory / model_filename_from_config

    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found in {run_directory}."); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading data for Feature Selection analysis from: {args.datafile_fs}")
    energy_idx_for_fs_data = run_config_loaded.get("energy_idx_used", project_config.ENERGY_IDX)
    blocks_for_analysis, _, _, _ = utils.read_file(
        args.datafile_fs, energy_idx_for_fs_data, project_config.PTABLE
    )
    
    g_mean_for_std, g_std_for_std = None, None
    if args.standardize_g:
        print(f"Calculating mean and std of g-vectors from '{args.datafile_fs}' for standardization...")
        g_mean_for_std, g_std_for_std = calculate_g_stats(blocks_for_analysis)
        if g_mean_for_std is None:
            print(f"Warning: Could not calculate g_stats. Standardization skipped."); args.standardize_g = False

    print(f"Loading trained model from: {model_path}")
    if model_type == "linear":
        loaded_model = models.AtomicLin(n_g_from_config, num_species_for_model)
    elif model_type == "mlp":
        loaded_model = models.AtomicMLP(n_g_from_config, num_species_for_model, hidden_units, depth_mlp)
    else:
        raise ValueError(f"Unknown model_type from config: {model_type}")

    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    print("Model loaded successfully.")

    feature_names = [f"SF-{i}" for i in range(n_g_from_config)]

    # Determine which species to analyze
    species_to_analyze = species_list_from_training if args.species is None else [sp for sp in args.species if sp in s2i_from_training]
    if not species_to_analyze:
        print("Error: None of the specified species are in the trained model. Exiting.")
        return
    print(f"Will analyze the following species: {species_to_analyze}")

    # --- Main Analysis Loop ---
    for target_species_symbol in species_to_analyze:
        species_idx = s2i_from_training[target_species_symbol]
        print(f"\n--- Analyzing Species: {target_species_symbol} (model sub-network index: {species_idx}) ---")

        sub_model = loaded_model.f[species_idx]
        sub_model.eval()
        importances_for_species = np.zeros(n_g_from_config)

        if args.method == "weights":
            if not args.standardize_g: print("WARNING: For 'weights' method, '--standardize_g' is recommended.")
            with torch.no_grad():
                if isinstance(sub_model, models.SpeciesMLP):
                    first_layer_weights = sub_model.net[0].weight.data.cpu().numpy()
                    importances_for_species = np.abs(first_layer_weights).sum(axis=0)
                elif isinstance(sub_model, torch.nn.Linear):
                    linear_weights = sub_model.weight.data.cpu().numpy()
                    importances_for_species = np.abs(linear_weights).squeeze()

        elif args.method == "pfi":
            print(f"Calculating PFI for {target_species_symbol}...")
            g_for_species_np = get_g_vectors_for_species(
                blocks_for_analysis, target_species_symbol, project_config.PTABLE,
                standardize=args.standardize_g, g_mean=g_mean_for_std, g_std=g_std_for_std
            )

            if g_for_species_np.shape[0] < 2:
                print(f"Not enough samples ({g_for_species_np.shape[0]}) for PFI. Skipping."); continue
            
            g_for_species_tensor = torch.from_numpy(g_for_species_np).to(device)
            with torch.no_grad():
                baseline_predictions = sub_model(g_for_species_tensor)
            
            pfi_scores = []
            for i in tqdm(range(n_g_from_config), desc=f"PFI for {target_species_symbol}", leave=False):
                temp_g_tensor = g_for_species_tensor.clone()
                perm_indices = torch.randperm(temp_g_tensor.size(0), device=device)
                temp_g_tensor[:, i] = temp_g_tensor[perm_indices, i]
                with torch.no_grad():
                    permuted_predictions = sub_model(temp_g_tensor)
                mae_increase = torch.abs(permuted_predictions - baseline_predictions).mean().item()
                pfi_scores.append(mae_increase)
            importances_for_species = np.array(pfi_scores)

        # Plot the primary importance bar chart
        if np.any(importances_for_species):
            plot_feature_importance(importances_for_species, feature_names, target_species_symbol, args.method, run_directory)
        else:
            print(f"No importances were calculated for {target_species_symbol}. Skipping.")
            continue
        
        # --- Evaluate performance vs. number of top-k features ---
        if args.eval_top_k:
            print(f"Performing top-k feature evaluation for {target_species_symbol}...")
            
            g_for_species_np = get_g_vectors_for_species(
                blocks_for_analysis, target_species_symbol, project_config.PTABLE,
                standardize=args.standardize_g, g_mean=g_mean_for_std, g_std=g_std_for_std
            )
            if g_for_species_np.shape[0] < 2:
                print(f"Not enough samples to run top-k evaluation. Skipping."); continue

            g_for_species_tensor = torch.from_numpy(g_for_species_np).to(device)
            
            sorted_indices = np.argsort(importances_for_species)[::-1].copy()
            
            # Run the evaluation, passing the scaling factor to get results in eV
            performance_data = evaluate_performance_by_feature_count(
                sub_model, g_for_species_tensor, sorted_indices, device, energy_std_for_scaling
            )

            if performance_data:
                k_vals, error_vals = zip(*performance_data)
                # Plot the results, also passing the scaling factor for correct labeling
                plot_performance_vs_feature_count(k_vals, error_vals, target_species_symbol, run_directory, energy_std_for_scaling)

if __name__ == "__main__":
    main_feature_selector()