# Part 2: Theoretical Background (Continued)

### 2.1.3 Element-Specific Neural Network Architecture

The neural network architecture processes symmetry functions to predict atomic energies. For a system containing multiple element types, separate networks are trained for each species. In the Lanthanum Zirconate system, three networks handle La, O, and Zr atoms respectively.

The AtomicMLP architecture employed in this study uses a feed-forward structure:

$$
\\mathbf{h}^{(0)} = \\mathbf{G}^i \\quad (\\text{input layer})
$$

$$
\\mathbf{h}^{(l+1)} = \\sigma(\\mathbf{W}^{(l)} \\mathbf{h}^{(l)} + \\mathbf{b}^{(l)}) \\quad (\\text{hidden layers})
$$

$$
E_{\\text{atom}}(i) = \\mathbf{w}^{\\text{out}} \\mathbf{h}^{(L)} + b^{\\text{out}} \\quad (\\text{output layer})
$$

The activation function $\\sigma$ is GELU (Gaussian Error Linear Unit), defined as:

$$
\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2}[1 + \\text{erf}(x/\\sqrt{2})]
$$

where $\\Phi(x)$ is the cumulative distribution function of the standard Gaussian. GELU provides smooth, non-monotonic activation with superior performance compared to ReLU for many deep learning tasks. The architecture used here employs 2 hidden layers with 256 units each (159 → 256 → 256 → 1).

### 2.2 Machine Learning Concepts

#### 2.2.1 Supervised Learning for Energy Prediction

Neural network potential training follows a supervised learning paradigm. High-quality reference data obtained from density functional theory (DFT) calculations provides ground truth atomic configurations and corresponding energies. The learning objective minimizes the discrepancy between predicted and reference energies across a training dataset.

The dataset comprises pairs $(\\mathbf{R}_n, E_n^{\\text{ref}})$ where $\\mathbf{R}_n$ represents the Cartesian coordinates of all atoms in configuration $n$, and $E_n^{\\text{ref}}$ is the quantum mechanical reference energy. During training, symmetry functions are computed for each atom, neural networks predict atomic energies, and the sum is compared against the reference.

#### 2.2.2 Training and Validation Splits

Proper machine learning practice requires partitioning data into separate subsets for training and validation. The training set (90% of data, 1,314 structures) is used to optimize network weights through backpropagation. The validation set (10%, 146 structures) evaluates model generalization on unseen configurations, preventing overfitting.

Data splitting employed random shuffling with a fixed seed (42) to ensure reproducibility. This stratified approach maintains representative sampling across the structural diversity present in the dataset. Validation performance serves as the primary metric for model selection and hyperparameter tuning decisions.

#### 2.2.3 Loss Function: Mean Absolute Error

The training objective minimizes Mean Absolute Error (MAE) rather than the more common Mean Squared Error (MSE). MAE offers several advantages for energy prediction:

$$
\\text{MAE} = \\frac{1}{N} \\sum_{n=1}^{N} |E_n^{\\text{pred}} - E_n^{\\text{ref}}|
$$

Unlike MSE, MAE treats all errors linearly rather than quadratically weighting outliers. This property is desirable for atomistic systems where occasional high-energy configurations (e.g., during rare event sampling) should not dominate the loss landscape. MAE values are also more interpretable, directly measuring average prediction error in units of energy (eV).

The implementation employs weighted MAE to account for dataset imbalances. Structures with less frequently observed energies receive higher weights, ensuring the model learns across the full energy distribution rather than overfitting to the most common configurations.

#### 2.2.4 Optimization: AdamW Algorithm

Network weights are optimized using AdamW, a variant of the Adam optimizer that decouples weight decay from the adaptive learning rate. AdamW maintains separate moving averages of gradients and squared gradients:

$$
m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t
$$

$$
v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2
$$

$$
\\theta_t = \\theta_{t-1} - \\alpha \\frac{m_t}{\\sqrt{v_t} + \\epsilon}
$$

The learning rate $\\alpha = 3 \\times 10^{-4}$ was selected through preliminary experimentation. Batch size of 1024 structures balances computational efficiency with gradient stability. Training proceeded for 300 epochs, taking approximately 3-4 minutes on an NVIDIA RTX 4060 GPU.

#### 2.2.5 Mixed Precision Training (AMP)

Automatic Mixed Precision (AMP) accelerates training by using 16-bit floating point (FP16) for most operations while maintaining 32-bit (FP32) precision for numerically sensitive calculations. This technique reduces memory consumption and leverages Tensor Cores available on modern GPUs, achieving 1.5-2× speedup without compromising accuracy.

The implementation uses PyTorch's built-in AMP with gradient scaling to prevent underflow in FP16 gradients. Loss values are computed in FP32 to maintain numerical stability, while forward passes through neural networks utilize mixed precision automatically.

### 2.3 Feature Importance Methods

#### 2.3.1 Motivation for Feature Selection

While the complete set of 159 symmetry functions provides comprehensive environmental description, several considerations motivate feature selection:

1. **Computational cost**: Calculating 159 descriptors for each atom introduces overhead during molecular dynamics
2. **Training efficiency**: Fewer features mean smaller input layers and faster gradient computation  
3. **Interpretability**: Identifying critical descriptors provides chemical insight into bonding interactions
4. **Overfitting prevention**: Dimensionality reduction can improve generalization by eliminating uninformative features

The challenge lies in identifying which subset of symmetry functions captures the essential physics while minimizing information loss.

#### 2.3.2 Weight-Based Importance

A naive approach to feature importance examines neural network weights connecting the input layer to the first hidden layer. Features with larger absolute weights ostensibly contribute more to predictions. For a network with input weights $\\mathbf{W}^{(0)} \\in \\mathbb{R}^{h \\times n_g}$, the importance of feature $j$ can be estimated as:

$$
I_j^{\\text{weights}} = \\sum_{k=1}^{h} |W_{kj}^{(0)}|
$$

However, this method suffers from several limitations. Network weights depend on feature scales, correlation between features confounds interpretation, and weight magnitude may not reflect actual contribution to final predictions due to nonlinear interactions across layers. Consequently, weight-based importance serves only as a preliminary estimate.

#### 2.3.3 Permutation Feature Importance (PFI)

Permutation Feature Importance offers a model-agnostic approach that directly measures each feature's contribution to predictive performance. The algorithm proceeds as follows:

1. **Baseline evaluation**: Compute validation error using the trained model with unperturbed features
2. **Feature permutation**: For each feature $j$, randomly shuffle its values across the validation set
3. **Degradation measurement**: Re-evaluate model performance with the permuted feature
4. **Importance calculation**: The increase in error quantifies feature importance

Mathematically, the importance of feature $j$ is:

$$
I_j^{\\text{PFI}} = \\text{Error}(\\text{model}, \\mathbf{X}^{\\pi_j}, \\mathbf{y}) - \\text{Error}(\\text{model}, \\mathbf{X}, \\mathbf{y})
$$

where $\\mathbf{X}^{\\pi_j}$ denotes the feature matrix with column $j$ permuted, and $\\mathbf{y}$ contains the reference energies. Permutation breaks the relationship between feature $j$ and the target, simulating the effect of removing that feature's information. Important features show large performance degradation when permuted; uninformative features cause negligible change.

