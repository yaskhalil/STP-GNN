# STP-GNN: Technical Foundations and Mathematical Logic

This document explains the core components of the STP-GNN framework in simple, rigorous terms. It breaks down the transformation of biological logic into a differentiable machine learning engine.

---

## 1. Algebraic State Space Representation (ASSR)

### What it is and why we need it
Standard biology models use "0" or "1" to show if a gene is off or on. However, computers cannot do calculus on single integers. We need to turn these "bits" into "vectors" so that a gene can be "80% on" or "20% off" during the learning process.

### The Math
For a single gene $x_i$:
- On ($1$): $\delta_2^1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
- Off ($0$): $\delta_2^2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

The global state $X$ for $n$ genes is:
$$X = x_1 \otimes x_2 \otimes \dots \otimes x_n$$

### Explanation
The symbol $\otimes$ is a Kronecker product. It creates a large vector that represents every possible combination of on/off states for all genes. If you have 10 genes, this vector has $2^{10} = 1024$ positions. Only one position is "1" at any time in a pure Boolean system.

### Output
A $2^n$ dimensional vector where a specific index represents the exact "snapshot" of the entire cell at that moment.

---

## 2. Temperature-Scaled Softmax

### What it is and why we need it
In a standard Boolean network, the rules are "hard" (e.g., if A is on, B MUST be off). This creates a "flat" landscape where the gradient (the slope) is zero. Without a slope, the AI cannot "climb" the mountain to find vulnerabilities. We use "Temperature" to melt these hard rules into soft curves.

### The Math
$$\tilde{M}_i = \frac{\exp(\tau \cdot \theta_i)}{\sum \exp(\tau \cdot \theta_i)}$$
Where $\tau$ (Tau) is our temperature (set to $10.0$).

### Explanation
As $\tau$ increases, the decision becomes sharper (more like a Boolean gate). By setting it to $10.0$, we keep the gate mostly logical but leave a tiny "slope" at the edges. This allows the backpropagation algorithm to "see" which way to nudge the parameters to break the system.

### Output
A "soft" transition matrix where instead of a $100\%$ chance of a gene turning on, it might be $99.9\%$, leaving $0.1\%$ for the AI to exploit.

---

## 3. Implicit STP (The Khatri-Rao Trick)

### What it is and why we need it
If we tried to create the full transition matrix for a 10-node network, it would have $1,024 \times 1,024$ entries ($1$ million). For $20$ nodes, it would be $1$ trillion. No computer has enough memory for this. We use "Implicit" math to calculate the result without ever building the giant matrix.

### The Math
Instead of $X_{t+1} = L \cdot X_t$, we calculate:
$$y_i = M_i \cdot X_t$$
$$X_{t+1} = y_1 \otimes y_2 \otimes \dots \otimes y_n$$

### Explanation
We calculate the next state of each gene *individually* first ($y_i$), and then combine them using the Kronecker product. This is mathematically identical to using the giant matrix $L$, but it only requires us to store the small local rules ($M_i$).

### Output
The state of the cell at the next time step ($t+1$) calculated using $1000\times$ less memory.

---

## 4. Adversarial Vulnerability Analysis (PGD)

### What it is and why we need it
Once the network is differentiable, we act as an "attacker." We want to find the smallest possible change to the biological rules that causes the cell cycle to stop. We use PGD (Projected Gradient Descent) to find these weak points.

### The Math
$$\theta_{new} = \theta_{old} + \epsilon \cdot \text{sign}(\nabla_{\theta} \text{Loss})$$

### Explanation
1. We define a "Loss" (e.g., the cell cycle is working).
2. We calculate the gradient ($\nabla$) to see which rule changes would most effectively *increase* that loss (break the cycle).
3. We "nudge" the rules in that direction by a tiny amount ($\epsilon$).

### Output
A list of "Topological Vulnerabilities"—the specific genes and conditions that, if perturbed, collapse the system.

---

## 5. Why p53 "Failed" vs. Cell Cycle Success

### The Comparison
- **p53 Model (5 nodes)**: This failed to show "Holographic" results because it was too simple. With only 5 genes, the "Kill Switch" was too obvious. There weren't enough backup pathways, so any attack worked. It lacked **Topological Complexity**.
- **Cell Cycle Model (10 nodes)**: This was the success. At 10 nodes, the network has enough redundancy that single-gene attacks often fail. This allowed us to prove that you need a **distributed attack** to break a complex system.

### Why Discrete Logic Fails the Attack
In a purely discrete simulation, you can only flip a gene from $0$ to $1$. There is no "in-between." This means you can't use gradients. You would have to try every single combination of flips (brute force), which is impossible for large networks.

---

## 6. DepMap Empirical Validation

### What it is and why we need it
We need to prove that the "weak points" the AI found in our math model actually exist in real human cancer cells. We use the Broad Institute's DepMap data.

### The Math (Stratification)
We take 1,000+ cell lines and group them:
- Group A: Genes with working Rb (inhibitor).
- Group B: Genes with broken Rb.
We then compare their dependency on E2F (the gene our AI attacked).

### Explanation
Our AI said: "The best way to break the cycle is to exploit the E2F bottleneck." 
The DepMap data showed: Cells that already have a broken Rb are *significantly more dependent* on E2F to survive. This means our AI found the exact same biological relationship that exists in real tumors.

### Output
Statistical parity: The AI's predicted "attack vector" matches the real-world "survival dependency" of cancer cells.
