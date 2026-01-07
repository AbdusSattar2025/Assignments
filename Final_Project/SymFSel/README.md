# criar o ambiente
conda env create -f environment.yml

# Activate
conda activate SymFSel

# Model Training

$ python train.py --datafile ./data/testing.data --model mlp --epochs 300 --val_ratio 0.1 --plot_set both --exp_prefix exp

# Feature Selection

$ python features.py --run_dir exp_res/ --datafile_fs ./data/testing.data --method pfi --eval_top_k

