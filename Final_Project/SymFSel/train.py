# train.py
import argparse
import torch
import pathlib
import numpy as np
import time
import json # Import for saving JSON

# Import project modules
import config
import models
import utils

# If Dataset, DataLoader, Subset are not being re-exported by utils, import them
from torch.utils.data import Dataset, DataLoader, Subset

# --- Local definitions of BlockDS and collate ---
# Move to utils.py if you want a cleaner train.py.
# For now, keeping here as in your last functional train.py.
class BlockDS(Dataset):
    def __init__(self, blocks, s2i, weights_tensor, device):
        self.b, self.i, self.w = blocks, s2i, weights_tensor
        self.device = device
        self.processed_blocks = []
        for idx, blk in enumerate(blocks): # Iterate over original blocks
            # Ensure that blk["atoms"] is in the format [(symbol, g_vector), ...]
            sym_np = np.fromiter((self.i[s] for s, _ in blk["atoms"]), np.int64)
            g_np = np.stack([g_val for _, g_val in blk["atoms"]], dtype=np.float32)
            
            self.processed_blocks.append({
                "sym": torch.from_numpy(sym_np).to(self.device),
                "g": torch.from_numpy(g_np).to(self.device),
                "E_tot": torch.tensor(blk["E_tot"], dtype=torch.float32).to(self.device),
                "weight": self.w[idx].to(self.device) # self.w is the full weights tensor
            })

        if self.device != "cpu" and len(self.processed_blocks) > 0:
            mem_sym_total = sum(item['sym'].nbytes for item in self.processed_blocks)
            mem_g_total = sum(item['g'].nbytes for item in self.processed_blocks)
            mem_etot_total = sum(item['E_tot'].nbytes for item in self.processed_blocks)
            # self.w is the full weights tensor, already on device
            mem_weights_full_tensor = self.w.nbytes if isinstance(self.w, torch.Tensor) else 0
            
            total_dataset_memory_bytes = mem_sym_total + mem_g_total + mem_etot_total + mem_weights_full_tensor
            
            unit = "MB"
            memory_val = total_dataset_memory_bytes / (1024**2)
            if memory_val > 1024:
                memory_val /= 1024
                unit = "GB"
            print(f"Full dataset tensors loaded to {self.device}. Estimated memory usage: {memory_val:.2f} {unit}")

    def __len__(self):
        return len(self.b) # Number of original blocks

    def __getitem__(self, idx):
        # Returns the pre-processed tensors for the given index
        blk_data = self.processed_blocks[idx]
        return blk_data["sym"], blk_data["g"], blk_data["E_tot"], blk_data["weight"]

def collate(batch):
    sym, g, e, w = zip(*batch) # These are already tensors on the correct device
    sym_cat = torch.cat(sym)
    g_cat = torch.cat(g)
    # Create blk_id, ensuring it's on the same device
    blk_id = torch.cat([torch.full((s.size(0),), i, dtype=torch.long, device=s.device)
                       for i, s in enumerate(sym)])
    return sym_cat, g_cat, blk_id, torch.stack(e), torch.stack(w)
# --- End of local BlockDS and collate definitions ---


def main():
    ap = argparse.ArgumentParser(description="Atomic Energy Predictor - Training Script")
    
    ap.add_argument("--datafile", default=str(config.DEFAULT_DATAFILE), help="Path to the data file.")
    ap.add_argument("--energy_idx", type=int, default=config.ENERGY_IDX, help="Index of energy value in the data file.")
    ap.add_argument("--model", choices=["linear","mlp"], default=config.DEFAULT_MODEL_TYPE, help="Model type: linear or mlp.")
    ap.add_argument("--hidden", type=int, default=config.DEFAULT_HIDDEN_UNITS, help="Number of hidden units in MLP layers.")
    ap.add_argument("--depth", type=int, default=config.DEFAULT_DEPTH, help="Number of hidden layers in MLP.")
    ap.add_argument("--lr", type=float, default=config.DEFAULT_LEARNING_RATE, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs.")
    ap.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size.")
    ap.add_argument("--workers",type=int, default=0, help="Number of DataLoader workers.")
    ap.add_argument("--val_ratio", type=float, default=config.DEFAULT_VALIDATION_RATIO, help="Fraction of data for validation.")
    ap.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    ap.add_argument("--no_weight",action="store_true", help="Disable inverse-frequency weighting for loss.")
    ap.add_argument("--plot_set", choices=["train","val","both"], default=config.DEFAULT_PLOT_SET, help="Dataset(s) for scatter plots.")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed.")
    ap.add_argument("--exp_prefix", type=str, default="", help="Prefix for saved figure and model filenames.")
    
    args = ap.parse_args()

    exp_prefix_str = args.exp_prefix
    if exp_prefix_str and not exp_prefix_str.endswith("_"):
        exp_prefix_str += "_"

    base_output_name = args.exp_prefix if args.exp_prefix else "run"
    current_output_dir = pathlib.Path(f"./{base_output_name}_res").resolve()
    current_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {current_output_dir}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dev}\n")

    blocks, n_g, species_list_from_data, energies_np = utils.read_file(
        args.datafile, args.energy_idx, config.PTABLE
    )
    if n_g is None:
        raise ValueError("Could not determine n_g (number of features) from datafile.")

    energy_mean = energies_np.mean()
    energy_std  = energies_np.std()
    for blk in blocks:
        blk["E_tot"] = (blk["E_tot"] - energy_mean) / energy_std
    energies_norm_np = (energies_np - energy_mean) / energy_std

    s2i = {s: i for i, s in enumerate(species_list_from_data)}

    if args.no_weight:
        weights_tensor = torch.ones(len(blocks), device=dev, dtype=torch.float32)
    else:
        weights_tensor = utils.make_weights(
            energies_norm_np,
            n_bins=getattr(config, 'WEIGHTING_N_BINS', 40), # Use getattr for default if not in config
            pow=getattr(config, 'WEIGHTING_POW', 1.0)
        ).to(dev)

    full_dataset = BlockDS(blocks, s2i, weights_tensor, dev)
    tr_idx, val_idx = utils.split_idx(len(full_dataset), args.val_ratio, args.seed)

    dl_tr = DataLoader(Subset(full_dataset, tr_idx), args.batch, shuffle=True,
                       collate_fn=collate, num_workers=args.workers,
                       pin_memory=False if dev == "cuda" else True,
                       prefetch_factor=4 if args.workers > 0 else None) # prefetch_factor if workers > 0
    dl_val = DataLoader(Subset(full_dataset, val_idx), args.batch, shuffle=False,
                        collate_fn=collate, num_workers=args.workers,
                        pin_memory=False if dev == "cuda" else True,
                        prefetch_factor=4 if args.workers > 0 else None)

    print(f"Initializing model on device: {dev}")
    num_species = len(species_list_from_data)
    if args.model == "linear":
        model = models.AtomicLin(n_g, num_species)
    else:
        model = models.AtomicMLP(n_g, num_species, args.hidden, args.depth)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), args.lr)
    amp_enabled = not args.no_amp and dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    loss_fn = torch.nn.L1Loss(reduction="none")

    print("Starting training...")
    training_start_time = time.perf_counter()
    epoch_numbers_history = []
    train_mae_history = []
    val_mae_history = []

    for ep in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        tr_mae_norm, _, _ = utils.run_epoch(dl_tr, len(tr_idx), model, opt, scaler, loss_fn, dev, amp_enabled, train=True)
        
        model.eval()
        val_mae_norm, _, _ = utils.run_epoch(dl_val, len(val_idx), model, None, scaler, loss_fn, dev, amp_enabled, train=False)
        
        epoch_duration = time.perf_counter() - epoch_start_time
        
        tr_mae = tr_mae_norm * energy_std
        val_mae = val_mae_norm * energy_std
        
        epoch_numbers_history.append(ep)
        # Ensure we are storing Python scalars for JSON and plotting
        train_mae_history.append(float(tr_mae) if isinstance(tr_mae, torch.Tensor) else tr_mae)
        val_mae_history.append(float(val_mae) if isinstance(val_mae, torch.Tensor) else val_mae)
        
        print(f"Epoch {ep:4d} | MAE Train {tr_mae:.6f} | MAE Val {val_mae:.6f} | Time: {epoch_duration:.2f}s")

    total_training_time = time.perf_counter() - training_start_time
    print(f"\nTraining finished. Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} min)")

    # --- Save Model ---
    # The model filename doesn't include the prefix twice, only the base run name
    model_save_filename = f"{base_output_name}_model.pth" 
    model_save_path = current_output_dir / model_save_filename
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Trained model state_dict saved to: {model_save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save model to {model_save_path}. Error: {e}")


    # --- Save Run Configuration ---
    run_config_to_save = {
        "model_path_relative": model_save_filename, # Save the model filename
        "datafile_used": args.datafile,
        "energy_idx_used": args.energy_idx,
        "model_type": args.model,
        "n_g": n_g,
        "species_list_ordered": species_list_from_data,
        "s2i_map_used": s2i,
        "hidden_units": args.hidden if args.model == "mlp" else None,
        "depth": args.depth if args.model == "mlp" else None,
        "learning_rate": args.lr,
        "epochs_trained": args.epochs, # Or ep if you want the actual number of completed epochs
        "batch_size": args.batch,
        "validation_ratio": args.val_ratio,
        "seed_used": args.seed,
        "amp_used": amp_enabled,
        "weighting_used": not args.no_weight,
        "energy_mean": float(energy_mean), # Convert to float for JSON serialization
        "energy_std": float(energy_std),   # Convert to float for JSON serialization
    }
    
    config_save_filename = f"{base_output_name}_run_config.json"
    config_save_path = current_output_dir / config_save_filename
    try:
        with open(config_save_path, 'w') as f:
            json.dump(run_config_to_save, f, indent=4)
        print(f"Run configuration successfully saved to: {config_save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save run configuration to {config_save_path}. Error: {e}")


    # --- Plot Results ---
    if epoch_numbers_history:
        loss_curve_filename_base = f"{base_output_name}_loss_curve.png"
        utils.plot_loss_curve(epoch_numbers_history, train_mae_history, val_mae_history,
                              filename=current_output_dir / loss_curve_filename_base)

    print("\nGenerating final predictions for scatter plots...")
    model.eval()
    _, p_tr_norm, r_tr_norm = utils.run_epoch(dl_tr, len(tr_idx), model, None, scaler, loss_fn, dev, amp_enabled, train=False)
    _, p_val_norm, r_val_norm = utils.run_epoch(dl_val, len(val_idx), model, None, scaler, loss_fn, dev, amp_enabled, train=False)

    p_tr = p_tr_norm * energy_std + energy_mean
    r_tr = r_tr_norm * energy_std + energy_mean
    p_val = p_val_norm * energy_std + energy_mean
    r_val = r_val_norm * energy_std + energy_mean

    if args.plot_set in ("train", "both"):
        train_scatter_filename_base = f"{base_output_name}_train_scatter.png"
        utils.scatter(r_tr.numpy(), p_tr.numpy(), "Train Set Predictions",
                      filename=current_output_dir / train_scatter_filename_base)
    if args.plot_set in ("val", "both"):
        val_scatter_filename_base = f"{base_output_name}_val_scatter.png"
        utils.scatter(r_val.numpy(), p_val.numpy(), "Validation Set Predictions",
                      filename=current_output_dir / val_scatter_filename_base)

if __name__ == "__main__":
    main()