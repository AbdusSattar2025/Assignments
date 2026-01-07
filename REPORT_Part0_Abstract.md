# Neural Network Potentials with Feature Selection for Lanthanum Zirconate Systems

**Author:** [Your Name]  
**Course:** [Course Code and Name]  
**Institution:** [University Name]  
**Date:** December 18, 2025

---

## Abstract

Neural network potentials represent a transformative approach in computational chemistry, offering quantum-mechanical accuracy at classical molecular dynamics speeds. This report presents a detailed investigation of feature selection methodologies applied to high-dimensional neural network potentials for Lanthanum Zirconate (La₂Zr₂O₇) systems. The primary objective was to understand and implement permutation feature importance analysis to identify the most critical symmetry functions among 159 candidate descriptors.

The study utilized the Behler-Parrinello framework, where atomic energies are predicted using element-specific neural networks trained on symmetry function descriptors. A dataset comprising 1,460 atomic configurations was analyzed using an AtomicMLP architecture with 256 hidden units and 2 layers. Training on 1,314 structures achieved validation R² of 0.801, demonstrating reasonable predictive capability. The implementation leverages modern deep learning optimizations including automatic mixed precision training and GPU acceleration.

Permutation feature importance analysis revealed distinct descriptor preferences across elements. For Lanthanum, radial symmetry function SF-22 exhibited the highest importance at 3.21%, while Oxygen showed preference for SF-22 (2.48%) and SF-70 (2.29%). Zirconium atoms demonstrated strongest sensitivity to SF-33 (2.95%) and angular functions. Top-k feature evaluation demonstrated that reducing the feature space from 159 to approximately 50-70 features maintains model performance while achieving computational savings of 55-68%.

The codebase analysis encompasses five core modules: configuration management, neural network architectures, utility functions, training pipeline, and feature analysis tools. Each component was systematically examined to understand the complete workflow from raw atomic coordinates to feature importance rankings. This work provides practical insights into dimensionality reduction for neural network potentials and establishes a foundation for future optimization of atomistic machine learning models.

**Keywords:** Neural network potentials, Feature selection, Permutation importance, Behler-Parrinello, Lanthanum zirconate, Symmetry functions, Deep learning, Computational chemistry

---
