import torch
from torch import nn
import numpy as np
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score 


def make_weights(energies, n_bins=40, pow=1.0):
    bins   = np.linspace(energies.min(), energies.max(), n_bins+1)
    ids    = np.digitize(energies, bins) - 1
    freq   = np.bincount(ids, minlength=n_bins) + 1e-6
    w      = (1.0 / freq[ids])**pow
    w_norm = w * (len(w)/w.sum())
    return torch.tensor(w_norm, dtype=torch.float32)

def split_idx(n, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed) 
    idx = rng.permutation(n)
    v = int(n * val_ratio)
    return idx[v:], idx[:v]

def z2sym(z, ptable_ref):
    return ptable_ref[z] if 0 < z < len(ptable_ref) and ptable_ref[z] else f"Z{z}"

def read_file(path, energy_idx, ptable_for_z2sym):
    print(f"DEBUG: Attempting to read file: {path}")
    blocks, species_set, n_g, energies = [], set(), None, []
    lines_read_count = 0
    try:
        with pathlib.Path(path).open() as f, tqdm(desc=f"Reading {pathlib.Path(path).name}") as bar:
            while True:
                line = f.readline()
                lines_read_count += 1
                if not line: # End of file
                    print(f"DEBUG: Reached end of file after {lines_read_count-1} lines.")
                    break 
                
                line_stripped = line.strip()
                if not line_stripped: # Skip empty lines
                    print(f"DEBUG: Skipping empty line {lines_read_count}.")
                    continue
                
                # print(f"DEBUG: Read line {lines_read_count} for 'm': '{line_stripped}'")
                try:
                    m = int(line_stripped)
                except ValueError:
                    print(f"ERROR_DEBUG: Could not parse 'm' (num_atoms) from line {lines_read_count}: '{line_stripped}'. Skipping block.")
                    # Try to skip to next potential 'm' by reading atom lines and energy line if they exist
                    # This is tricky; depends on how robust you want to be to malformed blocks
                    # For now, just continue, which might misinterpret next lines.
                    continue 

                # print(f"DEBUG: Parsed m = {m} for block.")
                atoms = []
                current_block_n_g_check = None # For consistency within the current block
                
                for i in range(m):
                    atom_line_num = lines_read_count + i + 1
                    atom_line = f.readline()
                    if not atom_line:
                        print(f"ERROR_DEBUG: Premature EOF at line {atom_line_num} while expecting atom data for block with m={m}.")
                        break # Break from inner atom loop

                    atom_line_stripped = atom_line.strip()
                    # print(f"DEBUG: Read line {atom_line_num} for atom: '{atom_line_stripped}'")
                    parts = atom_line_stripped.split()

                    if len(parts) < 2: # Must have at least Z and one g-feature
                        print(f"ERROR_DEBUG: Not enough parts in atom line {atom_line_num}: '{atom_line_stripped}'. Expected Z and g-features. Skipping atom.")
                        continue
                    
                    try:
                        z_val = int(float(parts[0]))
                        # Ensure there are g-features to parse before attempting np.array
                        if len(parts[1:]) == 0:
                            print(f"ERROR_DEBUG: No g-features found after Z in atom line {atom_line_num}: '{atom_line_stripped}'. Skipping atom.")
                            continue
                        g_features = np.array(parts[1:], float)
                    except ValueError:
                        print(f"ERROR_DEBUG: Could not parse Z or g-features from atom line {atom_line_num}: '{atom_line_stripped}'. Skipping atom.")
                        continue
                    
                    # print(f"DEBUG: Parsed g_features with length: {len(g_features)}")
                    if n_g is None: # First time setting n_g for the whole file
                        n_g = len(g_features)
                        current_block_n_g_check = n_g
                        print(f"DEBUG: n_g determined as {n_g} from first atom features at line {atom_line_num}.")
                    elif len(g_features) != n_g:
                        raise ValueError(
                            f"Inconsistent number of g_features in file {path} at line {atom_line_num}. "
                            f"Previously determined n_g={n_g}, but found {len(g_features)}. Atom line: '{atom_line_stripped}'"
                        )
                    
                    # Consistency check within the current block (if n_g was set from a *previous* block)
                    if current_block_n_g_check is None and n_g is not None: # n_g set by previous block, first atom of this block
                         current_block_n_g_check = len(g_features)
                         if current_block_n_g_check != n_g:
                             raise ValueError(
                                f"Inconsistent n_g for first atom of block at line {atom_line_num} compared to file's n_g. "
                                f"File n_g={n_g}, this atom has {current_block_n_g_check}. Atom line: '{atom_line_stripped}'"
                            )
                    elif current_block_n_g_check is not None and len(g_features) != current_block_n_g_check:
                         raise ValueError(
                            f"Inconsistent number of g_features *within the same block* in file {path} at line {atom_line_num}. "
                            f"Expected {current_block_n_g_check} based on first atom in this block, but found {len(g_features)}. Atom line: '{atom_line_stripped}'"
                        )


                    atom_symbol = z2sym(z_val, ptable_for_z2sym)
                    atoms.append((atom_symbol, g_features))
                    species_set.add(atom_symbol)
                
                lines_read_count += m # Advance line counter by number of atom lines read/skipped

                if m > 0 and not atoms:
                    print(f"WARNING_DEBUG: Block defined with m={m} but no valid atoms were successfully parsed from lines {lines_read_count-m+1} to {lines_read_count}.")
                
                energy_line_num = lines_read_count + 1
                energy_line = f.readline()
                if not energy_line:
                    print(f"ERROR_DEBUG: Premature EOF at line {energy_line_num} while expecting energy data for block.")
                    break # Break from outer while loop
                
                energy_line_stripped = energy_line.strip()
                # print(f"DEBUG: Read line {energy_line_num} for energy: '{energy_line_stripped}'")
                try:
                    energy_parts = energy_line_stripped.split()
                    if energy_idx >= len(energy_parts):
                        print(f"ERROR_DEBUG: energy_idx {energy_idx} is out of bounds for energy line {energy_line_num} with {len(energy_parts)} parts: '{energy_line_stripped}'. Skipping block.")
                        lines_read_count +=1 # Account for energy line
                        continue
                    e_tot = float(energy_parts[energy_idx])
                except (IndexError, ValueError) as e_parse_err: # Catch IndexError here too
                    print(f"ERROR_DEBUG: Could not parse energy from line {energy_line_num} using energy_idx={energy_idx}: '{energy_line_stripped}'. Error: {e_parse_err}. Skipping block.")
                    lines_read_count +=1 # Account for energy line
                    continue
                
                lines_read_count +=1 # Account for energy line successfully processed or skipped

                blocks.append({"atoms": atoms, "E_tot": e_tot})
                energies.append(e_tot)
                bar.update()

    except Exception as e:
        print(f"ERROR_DEBUG: An unexpected error occurred during file reading: {e}")
        # Optionally re-raise or handle more gracefully
        raise

    if n_g is None and blocks:
        print("WARNING_DEBUG: File processing finished. 'n_g' is still None, but some blocks were processed.")
        print("This implies that no atom features were successfully parsed to determine n_g.")
    elif n_g is None:
        print("WARNING_DEBUG: File processing finished. 'n_g' is None and no blocks were processed. The file might be empty or entirely unparsable.")

    print(f"DEBUG: read_file finished. n_g = {n_g}, num_blocks = {len(blocks)}, num_species = {len(species_set)}")
    return blocks, n_g, sorted(list(species_set)), np.asarray(energies)

def run_epoch(dl, idx_len, model, opt, scaler, loss_fn, dev, amp, train=True):
    # ... (código completo da função run_epoch como mostrado acima) ...
    context = torch.enable_grad if train else torch.inference_mode
    total_loss_sum, preds_list, reals_list = 0.0, [], []
    
    with context():
        for sym,g,blk,e_target,w in dl: 
            with torch.autocast("cuda",enabled=amp):
                pred = model(sym,g,blk,e_target.size(0))
                raw_error  = torch.abs(pred - e_target)
                loss = (raw_error * w).mean()
            
            if train:
                if opt is None:
                    raise ValueError("Optimizer (opt) cannot be None during training.")
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            total_loss_sum += loss.item() * e_target.size(0)
            preds_list.append(pred.cpu())
            reals_list.append(e_target.cpu())
    
    if idx_len == 0:
        return 0.0, torch.empty(0), torch.empty(0)

    all_preds = torch.cat(preds_list) if preds_list else torch.empty(0)
    all_reals = torch.cat(reals_list) if reals_list else torch.empty(0)
    
    return total_loss_sum / idx_len, all_preds, all_reals

def plot_loss_curve(epochs_ran, train_losses, val_losses, filename): # filename é o caminho completo
    plt.figure(figsize=(8,5))
    plt.plot(epochs_ran, train_losses, label="Training MAE", marker='o', linestyle='-')
    plt.plot(epochs_ran, val_losses, label="Validation MAE", marker='x', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename) # Salva diretamente no caminho fornecido
    plt.close()
    print(f"Saved loss curve to {filename}")


def scatter(reals, preds, title, filename): # filename é o caminho completo
    # Ensure reals and preds are numpy arrays for sklearn and min/max operations
    if not isinstance(reals, np.ndarray):
        reals = np.asarray(reals)
    if not isinstance(preds, np.ndarray):
        preds = np.asarray(preds)

    if reals.size == 0 or preds.size == 0:
        print(f"Warning: Cannot generate scatter plot for '{title}' due to empty data.")
        return np.nan # Return NaN for R² if data is empty

    r2 = r2_score(reals, preds)
    plt.figure(figsize=(5,5))
    plt.scatter(reals, preds, s=8, alpha=.6)
    
    lim_min = min(reals.min(), preds.min())
    lim_max = max(reals.max(), preds.max())
    lim = [lim_min, lim_max]

    # Handle case where all data points are identical (min == max)
    if lim_min == lim_max:
        delta = 0.1 * abs(lim_min) if abs(lim_min) > 1e-6 else 0.1
        lim = [lim_min - delta, lim_max + delta]
        if lim[0] == lim[1]: # If still identical (e.g. lim_min was 0)
             lim = [-0.1, 0.1]


    plt.plot(lim, lim, '--k')
    plt.xlabel("E_tot real")
    plt.ylabel("E_tot pred")
    plt.title(f"{title} – R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved scatter plot to {filename}")
    return r2
