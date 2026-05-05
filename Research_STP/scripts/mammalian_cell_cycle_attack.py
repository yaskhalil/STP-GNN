import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
from typing import List, Dict, Tuple

# =============================================================================
# I. OPTIMIZED STP OPERATOR (IMPLICIT VJP)
# =============================================================================

class ImplicitSTPFunction(torch.autograd.Function):
    """
    Optimized Vector-Jacobian Product for STP.
    Complexity: O(n * 2^n) time | O(2^n) memory.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, M_list: torch.Tensor) -> torch.Tensor:
        n = M_list.shape[0]
        y_list = [torch.mm(m, x) for m in M_list]
        x_next = y_list[0]
        for i in range(1, len(y_list)):
            x_next = torch.kron(x_next, y_list[i])
        ctx.save_for_backward(x, M_list, torch.stack(y_list))
        return x_next

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, M_list, y_list = ctx.saved_tensors
        n = M_list.shape[0]
        grad_x, grad_M = torch.zeros_like(x), torch.zeros_like(M_list)
        g_tensor = grad_output.view([2] * n)
        
        for i in range(n):
            y_comp = None
            for j in range(n):
                if i == j: continue
                y_comp = y_list[j] if y_comp is None else torch.kron(y_comp, y_list[j])
            
            permute_order = [i] + list(range(i)) + list(range(i+1, n))
            g_reshaped = g_tensor.permute(permute_order).reshape(2, -1)
            grad_y_i = torch.mm(g_reshaped, y_comp)
            grad_M[i] = torch.mm(grad_y_i, x.t())
            grad_x += torch.mm(M_list[i].t(), grad_y_i)
            
        return grad_x, grad_M

# =============================================================================
# II. NETWORK SPECIFICATION & LOGIC PARSING
# =============================================================================

class CellCycleIngestor:
    def __init__(self):
        self.node_order = ["CycD", "Rb", "E2F", "CycE", "CycA", "p27", "Cdc20", "Cdh1", "UbcH10", "CycB"]
        self.rules = {
            "CycD": "CycD",  # Input node
            "Rb": "(!CycD & !CycE & !CycA & !CycB) | (p27 & !CycD & !CycB)",
            "E2F": "(!Rb & !CycA & !CycB) | (p27 & !Rb & !CycB)",
            "CycE": "(E2F & !Rb)",
            "CycA": "(E2F | CycA) & !Rb & !Cdc20 & !(Cdh1 & UbcH10)",
            "p27": "(!CycD & !CycE & !CycA & !CycB) | (p27 & !(CycE & CycA) & !CycB & !CycD)",
            "Cdc20": "CycB",
            "Cdh1": "(!CycA & !CycB) | Cdc20 | (p27 & !CycB)",
            "UbcH10": "!Cdh1 | (Cdh1 & UbcH10 & (Cdc20 | CycA | CycB))",
            "CycB": "!Cdc20 & !Cdh1"
        }

    def evaluate(self, expr: str, state: Dict[str, bool]) -> bool:
        # Pre-process expressions
        e = expr.replace('!', ' not ').replace('&', ' and ').replace('|', ' or ')
        # Special case for CycD (Input)
        return bool(eval(e, {"__builtins__": None}, state))

    def generate_target_tensors(self):
        n = len(self.node_order)
        num_states = 2**n
        targets = torch.zeros((n, 2, num_states))
        for s in range(num_states):
            bits = {node: bool((s >> (n-1-i)) & 1) for i, node in enumerate(self.node_order)}
            for i, node in enumerate(self.node_order):
                res = self.evaluate(self.rules[node], bits)
                targets[i, 1 if res else 0, s] = 1.0
        return targets

# =============================================================================
# III. ADVERSARIAL ATTACK (PGD)
# =============================================================================

def perform_attack():
    print("="*80)
    print("MAMMALIAN CELL CYCLE: ADVERSARIAL VULNERABILITY ANALYSIS (N=10)")
    print("="*80)

    ingestor = CellCycleIngestor()
    n = len(ingestor.node_order)
    M_target = ingestor.generate_target_tensors()

    # 1. Base Model Training (Continuous Relaxation)
    # theta: (node, 2, 2^n)
    theta = nn.Parameter(torch.randn(n, 2, 2**n) * 0.1)
    optimizer = optim.Adam([theta], lr=0.2)
    tau = 10.0

    print("Step 1: Relaxing Boolean rules into differentiable logits...")
    for i in range(101):
        optimizer.zero_grad()
        M_current = torch.softmax(theta * tau, dim=1)
        loss = nn.MSELoss()(M_current, M_target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0: print(f"  Iteration {i:3d} | MSE Loss: {loss.item():.6f}")

    theta_base = theta.detach().clone()

    # 2. PGD Attack
    # Objective: Minimize the "Cycle Score" or maximize the distance from the cycle.
    # In a cell cycle, CycB is a marker of progression. 
    # We want to force the system into a "Broken Cycle" (e.g. Rb=1, p27=1, CycB=0).
    
    epsilon = 2.5
    theta_attack = theta_base.clone().requires_grad_(True)
    pgd_optimizer = optim.Adam([theta_attack], lr=0.1)

    # Initial state: G1 phase (CycD=1, Rb=1, p27=1, others=0)
    # State mapping: CycD(0), Rb(1), E2F(2), CycE(3), CycA(4), p27(5), Cdc20(6), Cdh1(7), UbcH10(8), CycB(9)
    # Bits: 1, 1, 0, 0, 0, 1, 0, 1, 0, 0  (Cdh1 is on in G1)
    init_state_idx = 0
    for i, node in enumerate(ingestor.node_order):
        val = 1 if node in ["CycD", "Rb", "p27", "Cdh1"] else 0
        init_state_idx |= (val << (n - 1 - i))
    
    x_init = torch.eye(2**n)[init_state_idx].unsqueeze(-1)

    print("\nStep 2: Executing PGD Attack (Projected Gradient Descent)...")
    for i in range(50):
        pgd_optimizer.zero_grad()
        x = x_init
        states = []
        for t in range(15):  # 15 steps of simulation
            x = ImplicitSTPFunction.apply(x, torch.softmax(theta_attack * tau, dim=1))
            x = x / (x.sum() + 1e-10)
            states.append(x)
        
        # Loss: We want to MAXIMIZE p27 and Rb (Inhibition) and MINIMIZE CycB (Progression)
        # CycB is node index 9. p27 is 5. Rb is 1.
        # State vector x is 2^10. We need to marginalize.
        
        terminal_x = states[-1]
        
        # Marginal for CycB (index 9)
        # CycB is ON if (s >> (10-1-9)) & 1 == 1, which is s & 1 == 1.
        p_cycB = terminal_x[1::2].sum()
        p_p27 = sum([terminal_x[s] for s in range(2**n) if (s >> (n-1-5)) & 1])
        
        # Attack Objective: Maximize CycB (Force mitosis) when it should be inhibited, 
        # or Break the cycle by forcing arrest. Let's aim for Cycle Arrest (High p27, Low CycB).
        # Actually, let's try to find the smallest perturbation that creates a NON-CYCLING state.
        # Variance of CycB over time should be low.
        
        cycB_trajectory = torch.stack([s[1::2].sum() for s in states])
        loss = -torch.var(cycB_trajectory) + p_cycB # Minimize variance AND CycB
        
        loss.backward()
        
        # PGD Step: Update and project onto epsilon ball
        with torch.no_grad():
            grad = theta_attack.grad
            theta_attack -= 0.1 * grad.sign()
            delta = torch.clamp(theta_attack - theta_base, -epsilon, epsilon)
            theta_attack.copy_(theta_base + delta)
        
    print(f"  PGD Complete. Minimal Structural Perturbation (epsilon={epsilon}) applied.")

    # 3. Mechanism of Action (MoA) Decoding
    diff = torch.abs(theta_attack - theta_base)
    top_vals, top_idx = torch.topk(diff.view(-1), 5)
    
    print("\n" + "="*80 + "\nMECHANISM OF ACTION (MoA) DISCOVERED\n" + "="*80)
    for rank, idx in enumerate(top_idx, 1):
        node_idx = (idx // (2 * (2**n))).item()
        row_idx = ((idx % (2 * (2**n))) // (2**n)).item()
        state_idx = (idx % (2**n)).item()
        
        node_name = ingestor.node_order[node_idx]
        row_label = "ON" if row_idx == 1 else "OFF"
        
        # Decode state_idx to parent values
        bits = [(state_idx >> (n-1-k)) & 1 for k in range(n)]
        context = ", ".join([f"{ingestor.node_order[k]}={bits[k]}" for k in range(n) if bits[k] == 1])
        
        p_base = torch.softmax(theta_base[node_idx, :, state_idx] * tau, dim=0)[row_idx].item()
        p_att  = torch.softmax(theta_attack[node_idx, :, state_idx] * tau, dim=0)[row_idx].item()
        
        print(f"RANK {rank}: {node_name} Transition Hijacked")
        print(f"  Logic Context: [{context}]")
        print(f"  Shift: P({node_name}={row_label}) {p_base:.4f} -> {p_att:.4f} [ΔP = {p_att - p_base:+.4f}]")

    # Final Result
    x_final = x_init
    for _ in range(20): x_final = ImplicitSTPFunction.apply(x_final, torch.softmax(theta_attack * tau, dim=1))
    p_cycB_final = x_final[1::2].sum().item()
    print(f"\nTerminal CycB Probability (Cycle Stability): {p_cycB_final:.4f}")
    if p_cycB_final < 0.1:
        print("VERDICT: CYCLE BROKEN. The attacker discovered a terminal G1/S arrest manifold.")
    else:
        print("VERDICT: CYCLE RESISTANT. High topological robustness detected.")

if __name__ == "__main__":
    perform_attack()
