# STP-GNN: Semi-Tensor Product Graph Neural Network

**STP-GNN** is a high-performance, differentiable adversarial engine designed for computational oncology and systems biology. It transforms discrete Boolean Gene Regulatory Networks (GRNs) into continuous algebraic manifolds, enabling the discovery of topological vulnerabilities through gradient-based optimization.

## Core Breakthroughs
- **Implicit STP Operator**: Bypasses the $O(4^n)$ state-space explosion, enabling million-state simulations (N=20) with $O(n \cdot 2^n)$ complexity using local Kronecker products.
- **Differentiable Logic**: Uses Temperature-Scaled Softmax ($\tau=10.0$) to melt rigid Boolean gates, solving the vanishing gradient problem in biological modeling.
- **Adversarial Vulnerability Discovery**: Implements Projected Gradient Descent (PGD) to identify "Holographic Resilience"—distributed vulnerabilities that single-node perturbations cannot trigger.
- **Empirical Parity**: Validated against human CRISPR-Cas9 dependency data (Broad Institute DepMap), achieving absolute statistical parity in identifying the Rb-E2F bottleneck.

## Repository Structure
- `Research_STP/scripts/`: The core STP-GNN engine, adversarial pipeline, and benchmarking tools.
- `Research_STP/outputs/`: (Local only) Visualization and result generation.
- `STP_GNN_EXPLAINER.md`: Technical foundations and mathematical derivations.

## Getting Started
### 1. Requirements
```bash
pip install torch numpy pandas matplotlib seaborn scipy
```

### 2. Run the Adversarial Pipeline
Execute the 10-node Mammalian Cell Cycle attack:
```bash
python Research_STP/scripts/mammalian_cell_cycle_attack.py
```

### 3. Empirical Validation
If you have the DepMap `CRISPRGeneEffect.csv` dataset, verify the predictions:
```bash
python Research_STP/scripts/cell_cycle_depmap_validation.py
```

## Mathematical Foundation
The global state updates algebraically via:
$$x(t+1) = \left( \bigotimes_{i=1}^n \tilde{M}_i \right) x(t)$$
By leveraging Vector-Jacobian Products (VJP), the engine calculates vulnerabilities without explicitly constructing the exponential transition matrix, making it thousands of times faster than traditional discrete simulators.

## License
MIT License
