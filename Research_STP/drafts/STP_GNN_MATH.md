# Mathematical Framework: Semi-Tensor Product GNNs (STP-GNN)

This document details the mathematical foundations of the STP-GNN research project, exploring the transition from traditional Boolean logic to differentiable, high-dimensional biological modeling.

## 1. Fundamentals of Semi-Tensor Product (STP)

The Semi-Tensor Product, denoted by $\ltimes$, is a generalization of the standard matrix product that allows for the multiplication of two matrices with incompatible dimensions, provided they satisfy a "factor" condition.

### The Left STP Definition
For two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$:
- If $n$ is a factor of $p$ ($p = n \cdot k$), the STP is defined as:
  $$A \ltimes B = (A \otimes I_k) B$$
- If $p$ is a factor of $n$ ($n = p \cdot k$), the STP is defined as:
  $$A \ltimes B = A (B \otimes I_k)$$

Where $\otimes$ denotes the **Kronecker Product**.

### Biological Context: Algebraic State Space Representation (ASSR)
In Gene Regulatory Networks (GRNs), a system of $n$ Boolean variables can be represented as a single state vector in a $2^n$-dimensional space using STP. 
A Boolean variable $x \in \{0, 1\}$ is mapped to a vector:
- $0 \to \delta_2^2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
- $1 \to \delta_2^1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

The state of the entire network is then:
$$x(t) = x_1(t) \ltimes x_2(t) \ltimes \dots \ltimes x_n(t)$$

## 2. The Differentiable STP Operator

To perform gradient-based learning or adversarial discovery, we must make the STP operation differentiable. We represent the logical transition of node $i$ as a matrix $M_i \in \mathbb{R}^{2 \times 2^n}$.

### Forward Pass
The global state update $x_{t+1}$ is the STP of the individual node transitions:
$$x_{t+1} = \bigltimes_{i=1}^n (M_i x_t)$$

In computational terms, this is implemented as a sequence of Kronecker products:
$$y_i = M_i x_t \in \mathbb{R}^2$$
$$x_{t+1} = y_1 \otimes y_2 \otimes \dots \otimes y_n \in \mathbb{R}^{2^n}$$

## 3. Gradient Optimization: From $O(4^n)$ to $O(n \cdot 2^n)$

### The Naive Jacobian Approach ($O(4^n)$)
Calculating the gradient of the loss $\mathcal{L}$ with respect to the transition matrices $M_i$ using standard Jacobian chain rules involves:
$$\frac{\partial \mathcal{L}}{\partial M_i} = \frac{\partial \mathcal{L}}{\partial x_{t+1}} \frac{\partial x_{t+1}}{\partial y_i} \frac{\partial y_i}{\partial M_i}$$

The term $\frac{\partial x_{t+1}}{\partial y_i}$ is a $2^n \times 2$ matrix. For a system with $n=20$ genes, the full Jacobian matrix has $2^{20} \times 2^{20}$ elements ($10^{12}$ elements), making it computationally impossible.

### The Optimized Vector-Jacobian Product (VJP)
We utilize the "Reshaping Trick" to avoid forming the full Jacobian. Given the gradient output $g = \frac{\partial \mathcal{L}}{\partial x_{t+1}}$, which we view as an $n$-dimensional tensor $G \in \mathbb{R}^{2 \times 2 \times \dots \times 2}$:

1.  **Partial Product Construction**: For each node $i$, we compute the Kronecker product of all *other* node states:
    $$\hat{y}_i = \bigotimes_{j \neq i} y_j$$
2.  **Tensor Permutation**: We permute the tensor $G$ such that the $i$-th dimension is first, then flatten the remaining $n-1$ dimensions.
3.  **Matrix Multiplication**: The gradient with respect to the intermediate output $y_i$ is:
    $$\nabla_{y_i} = \text{permute}(G, [i, \dots]) \cdot \hat{y}_i$$
4.  **Local Gradient**:
    $$\nabla_{M_i} = \nabla_{y_i} x_t^T$$

**Efficiency Gain:**
- **Jacobian Method**: $O(2^{2n})$ memory/time.
- **Optimized VJP**: $O(n \cdot 2^n)$ time, $O(2^n)$ memory.
*Numerical benchmarking (see `stp_gradient_benchmarking.py`) shows a 350x+ speedup for even small networks.*

## 4. Adversarial Discovery: The Kill Threshold ($\epsilon_{critical}$)

We define the "Kill Threshold" as the minimal structural perturbation $\epsilon$ to the network's logical weights $M$ that causes a "collapse" of the target attractor (e.g., forcing a cell into Apoptosis or breaking a stable cycle).

### Binary Critical Search
We use a binary search algorithm to find $\epsilon_{critical}$:
1. Define a perturbation: $M'_{i} = M_i - \epsilon \cdot \text{sign}(\nabla_{M_i})$
2. Simulate the network trajectories with $M'$.
3. Check if the target state (e.g., Apoptosis) is reached.
4. Adjust $\epsilon$ using binary search until the transition is found.

### Findings from the p53 Pipeline
- **$\epsilon_{critical}$ discovered: 2.9923**
- **Holographic Vulnerability**: Single-edge "knock-ins" of the most vulnerable connections fail to collapse the system. The "Kill" is a distributed property of the entire network topology, meaning causality is non-localized in these high-dimensional systems.
