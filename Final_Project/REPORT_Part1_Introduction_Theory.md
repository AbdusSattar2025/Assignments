# Part 1: Introduction and Theoretical Background

## 1. Introduction

### 1.1 Background: Neural Network Potentials in Computational Chemistry

Modern computational chemistry faces a fundamental challenge: accurately simulating atomic-scale phenomena requires quantum mechanical calculations, yet these methods remain prohibitively expensive for systems containing more than a few hundred atoms or for simulations extending beyond picoseconds. Traditional molecular dynamics employs empirical force fields that achieve computational efficiency through simplified analytical functions, but these approaches sacrifice the accuracy needed for studying complex chemical processes like bond breaking, surface reactions, or phase transitions.

Neural network potentials emerged as a transformative solution to this accuracy-efficiency dilemma. By learning the relationship between atomic configurations and energies from high-quality quantum mechanical reference data, machine learning models can interpolate across chemical space with near-quantum accuracy while maintaining computational costs comparable to classical force fields. This paradigm shift has opened new possibilities for studying materials properties, catalytic reactions, and condensed matter systems previously inaccessible to atomistic simulation.

### 1.2 The Behler-Parrinello Approach

In 2007, Jörg Behler and Michele Parrinello introduced a breakthrough framework that fundamentally changed how neural networks could represent potential energy surfaces. Their key innovation was recognizing that total energy could be decomposed into atomic contributions, with each atom's energy determined by its local chemical environment. This decomposition transforms the many-body problem into a collection of atomic-level predictions, making the approach scalable to systems of arbitrary size.

The Behler-Parrinello method introduced symmetry functions—mathematical descriptors that capture the local atomic environment while respecting fundamental physical symmetries. These descriptors ensure that the neural network predictions remain invariant under translation, rotation, and permutation of identical atoms. Radial symmetry functions encode distance information between atom pairs, while angular functions capture three-body correlations that describe local geometry and coordination patterns.

Element-specific neural networks form the architectural foundation of this approach. Rather than using a single network for all atom types, separate networks are trained for each chemical element, allowing the model to capture the unique electronic and geometric properties of different species. For a Lanthanum Zirconate system containing La, O, and Zr atoms, three independent networks learn element-specific energy contributions that sum to predict the total molecular energy.

### 1.3 Project Goal and Motivation

This project focuses on understanding feature selection within the context of neural network potentials for Lanthanum Zirconate (La₂Zr₂O₇), a pyrochlore-structured oxide with important applications in thermal barrier coatings and nuclear waste storage. While the Behler-Parrinello framework enables accurate energy predictions, the typical descriptor set comprises 159 symmetry functions per atom, creating computational overhead during both training and inference.

The central research question examines whether this high-dimensional feature space can be reduced without compromising prediction accuracy. Feature selection offers multiple benefits: reduced computational cost during descriptor calculation, faster neural network training, improved model interpretability, and potential mitigation of overfitting. Understanding which symmetry functions contribute most significantly to energy predictions provides chemical insights into the atomic-scale interactions governing material behavior.

Permutation feature importance emerged as the primary analysis technique for this investigation. Unlike weight-based methods that rely on network parameters, permutation importance directly measures each feature's contribution to model performance by systematically disrupting individual descriptors and quantifying the resulting prediction degradation. This model-agnostic approach provides reliable rankings even for complex nonlinear models.

### 1.4 Scope and Structure

This report presents a comprehensive analysis of the codebase implementing neural network potential training with feature selection capabilities. The investigation encompasses five core Python modules: configuration management (`config.py`), neural network architectures (`models.py`), utility functions (`utils.py`), training pipeline (`train.py`), and feature analysis (`features.py`). Each component receives detailed examination to understand the complete workflow from raw atomic coordinates to feature importance rankings.

Experimental validation was conducted using a dataset of 1,460 atomic configurations representing diverse structural arrangements of Lanthanum Zirconate. Training employed a 90/10 train-validation split with 300 epochs of optimization using the AdamW algorithm. Feature importance analysis identified the most critical descriptors for each element type and evaluated model performance as a function of feature count.

The remainder of this report is organized as follows: Section 2 establishes theoretical foundations for neural network potentials and feature importance methods. Section 3 provides detailed code architecture analysis, examining each module's design and implementation. Section 4 describes data formats and computational workflows. Section 5 discusses implementation optimizations including GPU acceleration and mixed precision training. Section 6 presents experimental results including training metrics and feature selection findings. Section 7 addresses limitations and future directions, while Section 8 concludes with key insights and recommendations.

---

## 2. Theoretical Background

### 2.1 High-Dimensional Neural Network Potentials (HDNNPs)

#### 2.1.1 Energy Decomposition

The fundamental assumption underlying Behler-Parrinello neural network potentials is that the total energy of a molecular system can be decomposed as a sum of atomic contributions:

$$
E_{\\text{total}} = \\sum_{i=1}^{N_{\\text{atoms}}} E_{\\text{atom}}(i)
$$

This decomposition transforms a many-body quantum mechanical problem into a collection of atomic-level machine learning tasks. Each atom's energy depends only on its local chemical environment, typically defined by a cutoff radius of 6-8 Ångströms. This locality assumption enables linear scaling with system size, making the approach applicable to large-scale molecular dynamics simulations.

The atomic energy $E_{\\text{atom}}(i)$ is predicted by an element-specific neural network that processes a vector of symmetry function descriptors $\\mathbf{G}^i$:

$$
E_{\\text{atom}}(i) = \\text{NN}_{Z_i}(\\mathbf{G}^i)
$$

where $Z_i$ denotes the atomic number (element type) of atom $i$, and $\\mathbf{G}^i \\in \\mathbb{R}^{n_g}$ is the descriptor vector containing $n_g$ symmetry function values. For the Lanthanum Zirconate system studied here, $n_g = 159$ symmetry functions describe each atomic environment.

#### 2.1.2 Symmetry Functions as Atomic Descriptors

Symmetry functions serve as the critical interface between atomic coordinates and neural network inputs. These mathematical descriptors must satisfy several fundamental requirements:

1. **Translational invariance**: Descriptors depend only on relative positions, not absolute coordinates
2. **Rotational invariance**: Values remain unchanged under coordinate system rotation  
3. **Permutational invariance**: Swapping identical atoms produces identical descriptors
4. **Smooth cutoff**: Descriptor values decay smoothly to zero beyond the cutoff radius

**Radial Symmetry Functions** capture two-body distance information between atom pairs:

$$
G_i^{\\text{rad}} = \\sum_{j \\neq i}^{\\text{neighbors}} e^{-\\eta(r_{ij}-R_s)^2} \\cdot f_c(r_{ij})
$$

where $r_{ij}$ is the distance between atoms $i$ and $j$, $\\eta$ controls the width of the Gaussian, $R_s$ shifts the center position, and $f_c(r_{ij})$ is a smooth cutoff function. Different combinations of $\\eta$ and $R_s$ parameters create a spectrum of radial descriptors sensitive to different distance ranges.

**Angular Symmetry Functions** encode three-body correlations that capture local geometric arrangements:

$$
G_i^{\\text{ang}} = 2^{1-\\zeta} \\sum_{j,k \\neq i} (1+\\lambda\\cos\\theta_{ijk})^\\zeta \\cdot e^{-\\eta(r_{ij}^2+r_{ik}^2+r_{jk}^2)} \\cdot f_c(r_{ij}) \\cdot f_c(r_{ik}) \\cdot f_c(r_{jk})
$$

The angle $\\theta_{ijk}$ describes the geometric arrangement of the triplet, while $\\zeta$ and $\\lambda$ control the angular resolution. These functions distinguish between linear, bent, and tetrahedral coordination geometries essential for describing chemical bonding.

The complete descriptor vector $\\mathbf{G}^i$ concatenates multiple radial and angular functions with varying parameters, creating a high-dimensional but comprehensive representation of the local atomic environment. For this project, 159 such functions describe each atom's surroundings.

