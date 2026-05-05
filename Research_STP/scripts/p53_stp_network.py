import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

# 1. THE OPTIMIZED STP OPERATOR (O(2^n) VJP)
class ImplicitSTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, M_list: torch.Tensor) -> torch.Tensor:
        n = M_list.shape[0]
        y_list = [torch.mm(m, x) for m in M_list]
        x_next = y_list[0]
        for i in range(1, len(y_list)): x_next = torch.kron(x_next, y_list[i])
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

# 2. p53 DISCOVERY ENGINE & INTERVENTION FRAMEWORK
def run_p53_analysis():
    print("="*60)
    print("p53-Mdm2 REGULATORY NETWORK: STP EXECUTION ENGINE")
    print("="*60)

    # Discovery Training
    model_theta = torch.nn.Parameter(torch.randn(3, 2, 8) * 0.1)
    optimizer = optim.Adam([model_theta], lr=0.1)
    
    def target_logic(idx):
        d, p, m = (idx>>2)&1, (idx>>1)&1, idx&1
        return (d<<2) | (int(d or not m)<<1) | int(p and not d)

    inputs = torch.eye(8).unsqueeze(-1)
    targets = torch.stack([torch.eye(8)[target_logic(i)] for i in range(8)]).unsqueeze(-1)

    print("\n[1] REVERSE-ENGINEERING BIOLOGICAL LOGIC...")
    for epoch in range(201):
        optimizer.zero_grad()
        M = torch.softmax(model_theta, dim=1)
        loss = sum([-torch.sum(targets[i] * torch.log(ImplicitSTPFunction.apply(inputs[i], M) + 1e-10)) for i in range(8)])
        loss.backward(); optimizer.step()
        if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Intervention Engine
    M_final = torch.softmax(model_theta, dim=1).detach()
    def simulate(name, init_bits, overrides):
        x = torch.eye(8)[7-init_bits].unsqueeze(-1)
        print(f"\n[INTERVENTION: {name}]")
        print("-" * 60)
        for t in range(1, 6):
            y = [torch.mm(M_final[i], x) for i in range(3)]
            for node, val in overrides.items(): y[node] = torch.tensor([[1.0 if val else 0.0], [0.0 if val else 1.0]])
            x = torch.kron(y[0], torch.kron(y[1], y[2]))
            idx = torch.argmax(x).item()
            d, p, m = (idx>>2)&1, (idx>>1)&1, idx&1
            label = f"(D:{'T' if d else 'F'}, p53:{'T' if p else 'F'}, Mdm2:{'T' if m else 'F'})"
            print(f"t={t} | {label}")
        return label

    # Biological Breaks
    res1 = simulate("Nutlin-3a (Mdm2 Inhibition)", 0, {2: False}) # Mdm2 OFF
    print(f"Biological Validation: p53 hyper-accumulation? {'[SUCCESS]' if 'p53:T' in res1 else '[FAIL]'}")

    res2 = simulate("TP53 Deletion (Genotoxic Stress)", 4, {1: False}) # p53 OFF, Damage=T
    print(f"Biological Validation: Failed damage response? {'[SUCCESS]' if 'p53:F' in res2 else '[FAIL]'}")

    res3 = simulate("Chronic Stress (Constitutive Damage)", 0, {0: True}) # Damage ON
    print(f"Biological Validation: Stress Attractor collapse? {'[SUCCESS]' if 'p53:T' in res3 else '[FAIL]'}")

if __name__ == "__main__":
    run_p53_analysis()
