import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import itertools
from typing import List, Dict, Tuple

class BiologicalLogicIngestor:
    def __init__(self, logic_rules: List[str]):
        self.raw_rules = logic_rules
        self.node_logic, self.node_parents, self.node_order = {}, {}, []
        self._parse_definitions()
    def _parse_definitions(self):
        pattern = r"Node\s+(\w+)\s+\{\s+logic\s+=\s+(.*);\s+\}"
        for rule in self.raw_rules:
            match = re.search(pattern, rule)
            if match:
                node_name = match.group(1); self.node_order.append(node_name)
                logic_expr = match.group(2); self.node_logic[node_name] = logic_expr
                self.node_parents[node_name] = sorted(list(set([p for p in re.findall(r'\b\w+\b', logic_expr) if not p.isdigit()])))
    def _evaluate_rule(self, expr, parent_states):
        python_expr = expr.replace('!', ' not ').replace('&', ' and ').replace('|', ' or ')
        return bool(eval(python_expr, {"__builtins__": None}, parent_states))
    def generate_global_target_tensors(self):
        n = len(self.node_order); num_states = 2**n; M_global = torch.zeros((n, 2, num_states))
        for state_idx in range(num_states):
            current_bits = {node: bool((state_idx >> (n - 1 - i)) & 1) for i, node in enumerate(self.node_order)}
            for i, node in enumerate(self.node_order):
                is_on = self._evaluate_rule(self.node_logic[node], current_bits)
                M_global[i, 1 if is_on else 0, state_idx] = 1.0
        return M_global

class ImplicitSTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M_list):
        n = M_list.shape[0]; y_list = [torch.mm(m, x) for m in M_list]; x_next = y_list[0]
        for i in range(1, len(y_list)): x_next = torch.kron(x_next, y_list[i])
        ctx.save_for_backward(x, M_list, torch.stack(y_list)); return x_next
    @staticmethod
    def backward(ctx, grad_output):
        x, M_list, y_list = ctx.saved_tensors; n = M_list.shape[0]; grad_x, grad_M = torch.zeros_like(x), torch.zeros_like(M_list)
        g_tensor = grad_output.view([2] * n)
        for i in range(n):
            y_comp = None
            for j in range(n):
                if i == j: continue
                y_comp = y_list[j] if y_comp is None else torch.kron(y_comp, y_list[j])
            permute_order = [i] + list(range(i)) + list(range(i+1, n))
            g_reshaped = g_tensor.permute(permute_order).reshape(2, -1)
            grad_y_i = torch.mm(g_reshaped, y_comp); grad_M[i] = torch.mm(grad_y_i, x.t()); grad_x += torch.mm(M_list[i].t(), grad_y_i)
        return grad_x, grad_M

class PGDAdversarialAttacker:
    def __init__(self, epsilon_max=0.15, alpha=0.02, iterations=25):
        self.epsilon_max, self.alpha, self.iterations = epsilon_max, alpha, iterations
    def attack(self, model_func, theta_base, x_init, target_idx, sim_steps=8):
        theta_adv = theta_base.clone().detach().requires_grad_(True)
        for k in range(self.iterations):
            x_curr = x_init
            for _ in range(sim_steps): x_curr = model_func(x_curr, torch.softmax(theta_adv, dim=1))
            prob_target = x_curr[target_idx]
            if prob_target.item() >= 0.95: break
            if theta_adv.grad is not None: theta_adv.grad.zero_()
            (-torch.log(prob_target + 1e-10)).backward()
            with torch.no_grad():
                theta_adv -= self.alpha * theta_adv.grad.sign()
                delta = torch.clamp(theta_adv - theta_base, -self.epsilon_max, self.epsilon_max)
                theta_adv.copy_(theta_base + delta)
        return theta_adv.detach()

class EpsilonCriticalSearch:
    def __init__(self, eps_min=0.01, eps_max=5.0, tolerance=0.01):
        self.eps_min, self.eps_max, self.tolerance = eps_min, eps_max, tolerance
    def find_critical_threshold(self, attacker_class, model_func, theta_base, x_init, target_idx, sim_steps=10):
        low, high = self.eps_min, self.eps_max; best_eps, best_theta = high, theta_base.clone()
        while (high - low) > self.tolerance:
            mid = (low + high) / 2
            attacker = attacker_class(epsilon_max=mid, alpha=mid/5.0, iterations=25)
            theta_candidate = attacker.attack(model_func, theta_base, x_init, target_idx, sim_steps)
            with torch.no_grad():
                x_c = x_init
                for _ in range(sim_steps): x_c = model_func(x_c, torch.softmax(theta_candidate, dim=1))
                success = x_c[target_idx].item() >= 0.95
            if success: best_eps, best_theta, high = mid, theta_candidate.clone(), mid
            else: low = mid
        print(f"CRITICAL DOSE DISCOVERED: ϵ_critical = {best_eps:.4f}"); return best_eps, best_theta

class AttractorCollapseVisualizer:
    def __init__(self, node_names): self.node_names = node_names
    def render(self, grad_matrix, theta_orig, theta_attacked, traj_mes, traj_apo, eps):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7)); sns.set_theme(style="whitegrid")
        sns.heatmap(torch.mean(torch.abs(grad_matrix), dim=1).view(len(self.node_names), -1).detach().numpy(), ax=axes[0], cmap="magma")
        t_o, t_a = theta_orig.view(-1).detach().numpy(), theta_attacked.view(-1).detach().numpy(); idx = np.argsort(np.abs(t_o - t_a))[-10:]
        axes[1].barh(np.arange(10)-0.2, t_o[idx], 0.4, label="Original", color="slategray"); axes[1].barh(np.arange(10)+0.2, t_a[idx], 0.4, label="Attacked", color="crimson"); axes[1].legend()
        axes[2].plot(traj_mes, label="P(Homeostatic)", color="slategray", lw=3, ls="--"); axes[2].plot(traj_apo, label="P(Apoptotic)", color="crimson", lw=4); axes[2].set_ylim(-0.05, 1.05); axes[2].legend()
        plt.tight_layout(); plt.savefig('critical_search_results.png')

def biological_edge_decoder(ingestor, theta_base, theta_attacked):
    n = len(ingestor.node_order); delta_theta = torch.abs(theta_attacked - theta_base)
    flat_delta = delta_theta.view(-1); top_vals, top_flat_indices = torch.topk(flat_delta, 5)
    print("\n" + "="*80 + "\nBIOLOGICAL ADVERSARIAL DECODING: TARGETED REGULATORY EDGES\n" + "="*80)
    for rank, flat_idx in enumerate(top_flat_indices, 1):
        node_idx = (flat_idx // (2 * (2**n))).item(); row_idx = ((flat_idx % (2 * (2**n))) // (2**n)).item(); state_idx = (flat_idx % (2**n)).item()
        target_node = ingestor.node_order[node_idx]; parents = ingestor.node_parents[target_node]; row_label = "ON" if row_idx == 1 else "OFF"
        global_bits = {name: (1 if (state_idx >> (n - 1 - k)) & 1 else 0) for k, name in enumerate(ingestor.node_order)}
        parent_config_str = ", ".join([f"{p}={global_bits[p]}" for p in parents])
        prob_base = torch.softmax(theta_base[node_idx, :, state_idx], dim=0)
        prob_attacked = torch.softmax(theta_attacked[node_idx, :, state_idx], dim=0)
        p_old, p_new = prob_base[row_idx].item(), prob_attacked[row_idx].item()
        print(f"\nRANK {rank}: HIGH-INFLUENCE VULNERABILITY\n  Target Node: {target_node}\n  Regulatory Context: {parent_config_str}\n  Logic Shift: P({target_node} = {row_label})\n  Magnitude: {p_old:.4f} ---> {p_new:.4f}  [ΔP = {p_new - p_old:+.4f}]")

def simulate_targeted_knockin(ingestor, theta_base, theta_attacked, target_idx, init_idx, steps=15):
    n = len(ingestor.node_order)
    delta_theta = torch.abs(theta_attacked - theta_base)
    flat_delta = delta_theta.view(-1)
    _, top_flat_indices = torch.topk(flat_delta, 5)
    theta_knockin = theta_base.clone().detach()
    flat_knockin = theta_knockin.view(-1)
    flat_attacked = theta_attacked.view(-1)
    for idx in top_flat_indices: flat_knockin[idx] = flat_attacked[idx]
    theta_knockin = flat_knockin.view(theta_base.shape)
    x_base = torch.eye(2**n)[init_idx].unsqueeze(-1)
    x_knockin = x_base.clone()
    p_apop_base, p_apop_knockin = [], []
    for t in range(steps):
        p_apop_base.append(x_base[target_idx].item())
        p_apop_knockin.append(x_knockin[target_idx].item())
        x_base = ImplicitSTPFunction.apply(x_base, torch.softmax(theta_base, dim=1))
        x_base = x_base / (x_base.sum() + 1e-10)
        x_knockin = ImplicitSTPFunction.apply(x_knockin, torch.softmax(theta_knockin, dim=1))
        x_knockin = x_knockin / (x_knockin.sum() + 1e-10)
    print(f"\n[Knock-in Validation] Terminal P(Apoptosis) @ T={steps}:")
    print(f"  Baseline Network: {p_apop_base[-1]:.4f}")
    print(f"  Top-5 Edge Knock-in: {p_apop_knockin[-1]:.4f}")
    if p_apop_knockin[-1] >= 0.90: print("  VERDICT: CAUSALITY CONFIRMED. The 5 edges are sufficient to shatter the attractor.")
    else: print("  VERDICT: INSUFFICIENT. The full network topology is required for the collapse.")

def run_pipeline():
    rules = ["Node p53_b1 { logic = (!p53_b2 & !Mdm2nuc) | (p53_b2); }", "Node p53_b2 { logic = (p53_b1 & !Mdm2nuc); }", "Node Mdm2cyt { logic = (p53_b1 & p53_b2); }", "Node Mdm2nuc { logic = (!p53_b1 & !Mdm2cyt & !DNAdam) | (!p53_b1 & Mdm2cyt) | (p53_b1 & Mdm2cyt); }", "Node DNAdam { logic = (!p53_b1 & DNAdam); }"]
    ingestor = BiologicalLogicIngestor(rules); M_target = ingestor.generate_global_target_tensors(); n = len(ingestor.node_order)
    theta = nn.Parameter(torch.randn(n, 2, 2**n) * 0.5); opt = optim.Adam([theta], lr=0.1)
    for _ in range(151): opt.zero_grad(); nn.MSELoss()(torch.softmax(theta, dim=1), M_target.to(theta.device)).backward(); opt.step()
    theta_base = theta.detach().clone()
    target_idx = sum([b << (n - 1 - i) for i, b in enumerate([1, 1, 0, 0, 1])]); init_idx = 0; x_init = torch.eye(2**n)[init_idx].unsqueeze(-1)
    search = EpsilonCriticalSearch(eps_min=0.01, eps_max=5.0, tolerance=0.01)
    eps_crit, theta_attacked = search.find_critical_threshold(PGDAdversarialAttacker, ImplicitSTPFunction.apply, theta_base, x_init, target_idx)
    t_f = theta_attacked.clone().requires_grad_(True); x_v = x_init
    for _ in range(10): x_v = ImplicitSTPFunction.apply(x_v, torch.softmax(t_f, dim=1))
    (-torch.log(x_v[target_idx] + 1e-10)).backward()
    tm, ta = [], []
    x_c = x_init
    for t in range(20):
        tm.append(x_c[init_idx].item()); ta.append(x_c[target_idx].item())
        x_c = ImplicitSTPFunction.apply(x_c, torch.softmax(theta_attacked, dim=1))
    AttractorCollapseVisualizer(ingestor.node_order).render(t_f.grad, theta_base, theta_attacked, tm, ta, eps_crit)
    biological_edge_decoder(ingestor, theta_base, theta_attacked)
    simulate_targeted_knockin(ingestor, theta_base, theta_attacked, target_idx, init_idx)

if __name__ == "__main__": run_pipeline()
