"""
Generate architecture diagrams for the SymFSel project.
Run this script to create PNG images for PowerPoint presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_project_architecture():
    """Create overall project architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'SymFSel Project Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Color scheme
    config_color = '#E8F4F8'  # Light blue
    model_color = '#FFE6CC'   # Light orange
    utils_color = '#E6F3E6'   # Light green
    script_color = '#F0E6FF'  # Light purple
    output_color = '#FFE6E6'  # Light pink
    
    # config.py
    config_box = FancyBboxPatch((0.5, 7), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=config_color, linewidth=2)
    ax.add_patch(config_box)
    ax.text(1.5, 7.5, 'config.py', ha='center', va='center', fontweight='bold')
    ax.text(1.5, 7.2, '• Hyperparameters\n• Paths\n• Constants', 
            ha='center', va='center', fontsize=8)
    
    # models.py
    model_box = FancyBboxPatch((4, 7), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=model_color, linewidth=2)
    ax.add_patch(model_box)
    ax.text(5.25, 7.5, 'models.py', ha='center', va='center', fontweight='bold')
    ax.text(5.25, 7.2, '• SpeciesMLP\n• AtomicMLP\n• AtomicLin', 
            ha='center', va='center', fontsize=8)
    
    # utils.py
    utils_box = FancyBboxPatch((7.5, 7), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=utils_color, linewidth=2)
    ax.add_patch(utils_box)
    ax.text(8.5, 7.5, 'utils.py', ha='center', va='center', fontweight='bold')
    ax.text(8.5, 7.2, '• read_file\n• run_epoch\n• plotting', 
            ha='center', va='center', fontsize=8)
    
    # train.py
    train_box = FancyBboxPatch((1, 4.5), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=script_color, linewidth=2)
    ax.add_patch(train_box)
    ax.text(2.5, 5.5, 'train.py', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(2.5, 5.1, '1. Load & preprocess data', ha='center', va='center', fontsize=8)
    ax.text(2.5, 4.9, '2. Create AtomicMLP model', ha='center', va='center', fontsize=8)
    ax.text(2.5, 4.7, '3. Train with AdamW + AMP', ha='center', va='center', fontsize=8)
    
    # features.py
    features_box = FancyBboxPatch((6, 4.5), 3, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='black', facecolor=script_color, linewidth=2)
    ax.add_patch(features_box)
    ax.text(7.5, 5.5, 'features.py', ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(7.5, 5.1, '1. Load trained model', ha='center', va='center', fontsize=8)
    ax.text(7.5, 4.9, '2. Compute PFI scores', ha='center', va='center', fontsize=8)
    ax.text(7.5, 4.7, '3. Generate importance plots', ha='center', va='center', fontsize=8)
    
    # Data input
    data_box = FancyBboxPatch((1.5, 2), 2, 0.8, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor='#FFF9E6', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(2.5, 2.4, 'testing.data', ha='center', va='center', fontweight='bold')
    ax.text(2.5, 2.1, '(1460 structures)', ha='center', va='center', fontsize=8)
    
    # Output directory
    output_box = FancyBboxPatch((3.5, 0.5), 3, 1.2, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.4, 'exp_res/', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(5, 1.1, '• exp_model.pth', ha='center', va='center', fontsize=8)
    ax.text(5, 0.9, '• exp_run_config.json', ha='center', va='center', fontsize=8)
    ax.text(5, 0.7, '• features_*_pfi_importance.png', ha='center', va='center', fontsize=8)
    
    # Arrows - config to scripts
    arrow1 = FancyArrowPatch((1.5, 7), (2, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='blue')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((1.5, 7), (7, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='blue')
    ax.add_patch(arrow2)
    
    # Arrows - models to scripts
    arrow3 = FancyArrowPatch((5, 7), (3, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='orange')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((5.5, 7), (7, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='orange')
    ax.add_patch(arrow4)
    
    # Arrows - utils to scripts
    arrow5 = FancyArrowPatch((8.5, 7), (3.5, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='green')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((8.5, 7), (8, 6), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=1.5, color='green')
    ax.add_patch(arrow6)
    
    # Arrows - data flow
    arrow7 = FancyArrowPatch((2.5, 2.8), (2.5, 4.5), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow7)
    
    # Arrows - train to output
    arrow8 = FancyArrowPatch((2.5, 4.5), (4, 1.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple')
    ax.add_patch(arrow8)
    
    # Arrows - features to output
    arrow9 = FancyArrowPatch((7.5, 4.5), (6.5, 1.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='purple')
    ax.add_patch(arrow9)
    
    # Arrow - train to features (model checkpoint)
    arrow10 = FancyArrowPatch((4, 5.25), (6, 5.25), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='red', linestyle='--')
    ax.add_patch(arrow10)
    ax.text(5, 5.5, 'model checkpoint', ha='center', va='bottom', 
            fontsize=8, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig('diagram1_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Created: diagram1_architecture.png")
    plt.close()


def create_model_architecture():
    """Create detailed model architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'AtomicMLP Neural Network Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((3.5, 8), 3, 0.6, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='black', facecolor='#E8F4F8', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 8.3, 'Input: Atoms + Symmetry Functions', 
            ha='center', va='center', fontweight='bold')
    
    # Three species networks
    species_y = [6.5, 4.5, 2.5]
    species_names = ['Lanthanum (La)', 'Oxygen (O)', 'Zirconium (Zr)']
    colors = ['#FFE6CC', '#E6F3E6', '#F0E6FF']
    
    for i, (y_pos, name, color) in enumerate(zip(species_y, species_names, colors)):
        # Species box
        species_box = FancyBboxPatch((0.5, y_pos - 0.3), 2, 0.6, 
                                     boxstyle="round,pad=0.05", 
                                     edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(species_box)
        ax.text(1.5, y_pos, name, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # SpeciesMLP layers
        # Input to hidden
        layer1_box = FancyBboxPatch((3, y_pos - 0.25), 1.5, 0.5, 
                                    boxstyle="round,pad=0.03", 
                                    edgecolor='black', facecolor='#FFEEAA', linewidth=1)
        ax.add_patch(layer1_box)
        ax.text(3.75, y_pos, '159 → 256\n+ GELU', ha='center', va='center', fontsize=7)
        
        # Hidden to hidden
        layer2_box = FancyBboxPatch((5, y_pos - 0.25), 1.5, 0.5, 
                                    boxstyle="round,pad=0.03", 
                                    edgecolor='black', facecolor='#FFEEAA', linewidth=1)
        ax.add_patch(layer2_box)
        ax.text(5.75, y_pos, '256 → 256\n+ GELU', ha='center', va='center', fontsize=7)
        
        # Output layer
        layer3_box = FancyBboxPatch((7, y_pos - 0.25), 1.5, 0.5, 
                                    boxstyle="round,pad=0.03", 
                                    edgecolor='black', facecolor='#FFEEAA', linewidth=1)
        ax.add_patch(layer3_box)
        ax.text(7.75, y_pos, '256 → 1\n(Energy)', ha='center', va='center', fontsize=7)
        
        # Arrows between layers
        ax.arrow(2.5, y_pos, 0.45, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
        ax.arrow(4.5, y_pos, 0.45, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
        ax.arrow(6.5, y_pos, 0.45, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
        
        # Arrow from input to species
        ax.arrow(5, 7.7, -3.3, y_pos - 7.4, head_width=0.1, head_length=0.05, 
                fc='gray', ec='gray', alpha=0.5, linestyle='--')
    
    # Sum energies
    sum_box = FancyBboxPatch((8, 4), 1.5, 1, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor='#FFE6E6', linewidth=2)
    ax.add_patch(sum_box)
    ax.text(8.75, 4.5, '∑\nTotal\nEnergy', ha='center', va='center', 
            fontweight='bold', fontsize=10)
    
    # Arrows to sum
    for y_pos in species_y:
        ax.arrow(8.5, y_pos, 0.2, 4.5 - y_pos - 0.1, 
                head_width=0.1, head_length=0.05, fc='red', ec='red')
    
    # Legend
    legend_y = 0.8
    ax.text(1, legend_y, 'Key Features:', fontweight='bold', fontsize=9)
    ax.text(1, legend_y - 0.3, '• Separate network per element', fontsize=8)
    ax.text(1, legend_y - 0.5, '• 2 hidden layers (256 units each)', fontsize=8)
    ax.text(1, legend_y - 0.7, '• GELU activation function', fontsize=8)
    ax.text(1, legend_y - 0.9, '• Element-specific energy prediction', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('diagram2_model_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Created: diagram2_model_architecture.png")
    plt.close()


def create_training_workflow():
    """Create training workflow diagram"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Training Workflow (train.py)', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    steps = [
        ('testing.data\n(1460 structures)', 10.5, '#FFF9E6'),
        ('Parse Data\nutils.read_file()', 9.5, '#E6F3E6'),
        ('Normalize Energies\nE_norm = (E - μ) / σ', 8.5, '#E6F3E6'),
        ('Create Dataset\nBlockDS + DataLoader', 7.5, '#E6F3E6'),
        ('Initialize Model\nAtomicMLP(n_g=159, ...)', 6.5, '#FFE6CC'),
        ('Setup Optimizer\nAdamW(lr=3e-4)', 5.5, '#F0E6FF'),
        ('Training Loop\n300 epochs\nwith AMP', 4.0, '#F0E6FF'),
        ('Save Checkpoint\nexp_model.pth', 2.5, '#FFE6E6'),
        ('Generate Plots\nloss curves & scatter', 1.5, '#FFE6E6'),
    ]
    
    for i, (text, y, color) in enumerate(steps):
        if i == 6:  # Training loop is taller
            height = 1.2
        else:
            height = 0.7
            
        box = FancyBboxPatch((2, y - height/2), 6, height, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(5, y, text, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Arrow to next step
        if i < len(steps) - 1:
            next_y = steps[i+1][1]
            if i == 5:
                arrow_start_y = y - height/2
                arrow_end_y = next_y + (1.2 if i+1 == 6 else 0.7)/2
            else:
                arrow_start_y = y - height/2
                arrow_end_y = next_y + (1.2 if i+1 == 6 else 0.7)/2
            
            ax.arrow(5, arrow_start_y, 0, arrow_end_y - arrow_start_y + 0.1, 
                    head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Add timing info
    ax.text(8.5, 4.0, '~3-4 min\non RTX 4060', 
            ha='left', va='center', fontsize=9, style='italic', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('diagram3_training_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Created: diagram3_training_workflow.png")
    plt.close()


def create_pfi_workflow():
    """Create PFI feature importance workflow"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Permutation Feature Importance (PFI)', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Load model
    box1 = FancyBboxPatch((2.5, 8), 5, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#FFE6CC', linewidth=2)
    ax.add_patch(box1)
    ax.text(5, 8.35, 'Load Trained Model (exp_model.pth)', 
            ha='center', va='center', fontweight='bold')
    
    # For each species
    box2 = FancyBboxPatch((2.5, 6.8), 5, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#E6F3E6', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 7.15, 'For each species: La, O, Zr', 
            ha='center', va='center', fontweight='bold')
    
    # Get baseline
    box3 = FancyBboxPatch((2.5, 5.6), 5, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#F0E6FF', linewidth=2)
    ax.add_patch(box3)
    ax.text(5, 5.95, '1. Get Baseline Predictions (all 159 features)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # PFI loop
    loop_box = FancyBboxPatch((1.5, 2.5), 7, 2.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='red', facecolor='#FFF9E6', linewidth=2.5,
                              linestyle='--')
    ax.add_patch(loop_box)
    ax.text(2, 4.8, 'For i = 1 to 159:', fontweight='bold', fontsize=10, color='red')
    
    # Steps inside loop
    ax.text(5, 4.3, '2a. Shuffle feature i values', ha='center', va='center', fontsize=9)
    ax.text(5, 3.9, '2b. Get new predictions', ha='center', va='center', fontsize=9)
    ax.text(5, 3.5, '2c. Measure MAE increase', ha='center', va='center', fontsize=9)
    ax.text(5, 3.1, '2d. Store importance score', ha='center', va='center', fontsize=9)
    ax.text(5, 2.7, 'importance[i] = MAE_permuted - MAE_baseline', 
            ha='center', va='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    
    # Rank features
    box4 = FancyBboxPatch((2.5, 1.3), 5, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#E6F3E6', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 1.65, '3. Rank Features by Importance Score', 
            ha='center', va='center', fontweight='bold')
    
    # Output
    box5 = FancyBboxPatch((2.5, 0.2), 5, 0.7, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor='#FFE6E6', linewidth=2)
    ax.add_patch(box5)
    ax.text(5, 0.55, 'Output: features_<element>_pfi_importance.png', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Arrows
    ax.arrow(5, 7.8, 0, -0.25, head_width=0.2, head_length=0.05, fc='black', ec='black', linewidth=1.5)
    ax.arrow(5, 6.8, 0, -0.45, head_width=0.2, head_length=0.05, fc='black', ec='black', linewidth=1.5)
    ax.arrow(5, 5.3, 0, -0.25, head_width=0.2, head_length=0.05, fc='black', ec='black', linewidth=1.5)
    ax.arrow(5, 2.2, 0, -0.25, head_width=0.2, head_length=0.05, fc='black', ec='black', linewidth=1.5)
    ax.arrow(5, 1.3, 0, -0.35, head_width=0.2, head_length=0.05, fc='black', ec='black', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig('diagram4_pfi_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Created: diagram4_pfi_workflow.png")
    plt.close()


if __name__ == "__main__":
    print("Generating architecture diagrams...")
    print("-" * 50)
    
    create_project_architecture()
    create_model_architecture()
    create_training_workflow()
    create_pfi_workflow()
    
    print("-" * 50)
    print("\n✅ All diagrams created successfully!")
    print("\nGenerated files:")
    print("  1. diagram1_architecture.png - Overall project architecture")
    print("  2. diagram2_model_architecture.png - Neural network details")
    print("  3. diagram3_training_workflow.png - Training pipeline")
    print("  4. diagram4_pfi_workflow.png - Feature importance process")
    print("\nYou can now insert these PNG files into your PowerPoint!")
