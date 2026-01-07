# config.py

import pathlib

# --- Paths ---
DATA_DIR = pathlib.Path("./data") # Example base data directory
OUTPUT_DIR_BASE = pathlib.Path("./results")
DEFAULT_DATAFILE = DATA_DIR / "blocos_real.dat"

# --- Data Parameters ---
ENERGY_IDX = 2

# --- Model Architecture Defaults (can be overridden by argparse) ---
DEFAULT_MODEL_TYPE = "mlp"
DEFAULT_HIDDEN_UNITS = 256
DEFAULT_DEPTH = 2

# --- Training Hyperparameters Defaults ---
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 1024
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_SEED = 42

# --- Feature Engineering / Preprocessing ---
NORMALIZE_TARGET_ENERGIES = False
USE_INVERSE_FREQUENCY_WEIGHTING = False
WEIGHTING_N_BINS = 40
WEIGHTING_POW = 1.0

# --- Plotting / Output ---
DEFAULT_PLOT_SET = "both" # 'train', 'val', 'both'

# --- Periodic Table (could also be in a utils.py) ---
PTABLE = ["","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si",
          "P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni",
          "Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb",
          "Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
          "Cs","Ba","La","Ce","Pr","Nd"]

# --- Device ---
PREFERRED_DEVICE = "cuda"