import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import Dict, List, Tuple

class AttractorCollapseVisualizer:
    def __init__(self, node_names: List[str]):
        self.node_names = node_names
        self.n = len(node_names)
        sns.set_theme(style="white", context="talk")

    def plot_complete_attack_analysis(
        self, 
        theta_orig: torch.Tensor, 
        grad_matrix: torch.Tensor, 
        epsilon: float,
        trajectories: Dict[str, List[float]]
    ):
        fig, axes = plt.subplots(1, 3, figsize=(28, 8))
        
        # --- Visualization 1: Gradient Topography ---
        ax = axes[0]
        abs_grad = torch.abs(grad_matrix).cpu().numpy()
        sns.heatmap(abs_grad, ax=ax, cmap="magma", cbar_kws={'label': '|Grad|'})
        
        flat_indices = np.argsort(abs_grad.flatten())[-5:]
        for idx in flat_indices:
            row, col = divmod(idx, self.n)
            ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='cyan', lw=3))

        ax.set_title("1. Gradient Topography (Vulnerability Map)")
        ax.set_xticklabels(self.node_names, rotation=90)
        ax.set_yticklabels(self.node_names, rotation=0)

        # --- Visualization 2: Fractional Shift ---
        ax = axes[1]
        top_k = 10
        top_k_indices = np.argsort(abs_grad.flatten())[-top_k:]
        
        edge_labels, orig_vals, shift_vals = [], [], []
        delta_theta = -epsilon * torch.sign(grad_matrix)
        theta_attacked = theta_orig + delta_theta

        for idx in top_k_indices:
            r, c = divmod(idx, self.n)
            edge_labels.append(f"{self.node_names[c]}->{self.node_names[r]}")
            orig_vals.append(theta_orig[r, c].item())
            shift_vals.append(theta_attacked[r, c].item())

        y = np.arange(top_k)
        ax.barh(y - 0.2, orig_vals, 0.4, label="Original", color="slategray", alpha=0.6)
        ax.barh(y + 0.2, shift_vals, 0.4, label="Attacked", color="crimson")
        ax.set_yticks(y)
        ax.set_yticklabels(edge_labels)
        ax.set_title("2. Fractional Shift (Micro-Cuts)")
        ax.legend()

        # --- Visualization 3: Attractor Collapse Trajectory ---
        ax = axes[2]
        time_steps = np.arange(len(trajectories['mesenchymal']))
        ax.plot(time_steps, trajectories['mesenchymal'], label="P(Mesenchymal)", color="slategray", lw=4, ls='--')
        ax.plot(time_steps, trajectories['apoptotic'], label="P(Apoptotic)", color="crimson", lw=5)
        ax.fill_between(time_steps, trajectories['mesenchymal'], alpha=0.1, color="slategray")
        ax.fill_between(time_steps, trajectories['apoptotic'], alpha=0.1, color="crimson")
        ax.set_title("3. Attractor Collapse Trajectory")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('attractor_collapse.png')
        print("Visualization saved to attractor_collapse.png")

if __name__ == "__main__":
    nodes = ["TGFB", "SNAI1", "ZEB1", "miR200", "CDH1", "VIM", "AKT", "ERK", 
             "p53", "MDM2", "BCL2", "BAX", "Casp3", "FLI1", "STAT3", "IL6"]
    visualizer = AttractorCollapseVisualizer(nodes)
    mock_theta = torch.ones(16, 16) * 2.5
    mock_grads = torch.randn(16, 16)
    t_steps = 20
    mes_prob = np.exp(-np.linspace(0, 5, t_steps))
    apo_prob = 1 - mes_prob
    visualizer.plot_complete_attack_analysis(mock_theta, mock_grads, 0.02, 
        {'mesenchymal': mes_prob.tolist(), 'apoptotic': apo_prob.tolist()})
