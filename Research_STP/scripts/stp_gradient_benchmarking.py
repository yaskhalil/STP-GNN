import torch
import time

def forward_pass(x, W1, W2, W3, use_relu=False):
    a1 = torch.matmul(W1, x)
    z1 = torch.relu(a1) if use_relu else a1
    a2 = torch.matmul(W2, z1)
    z2 = torch.relu(a2) if use_relu else a2
    z3 = torch.matmul(W3, z2)
    return a1, z1, a2, z2, z3

def backward_jacobian(delta, x, z1, z2, W2, W3, a1, a2, use_relu=False):
    d = z1.shape[0]
    n = x.shape[0]
    
    # Sigmoid/ReLU masks
    s2 = (a2 > 0).float() if use_relu else torch.ones_like(a2)
    s1 = (a1 > 0).float() if use_relu else torch.ones_like(a1)
    
    # J3 = delta * z2^T
    grad_W3 = delta * z2.t()
    
    # J2 using Kronecker: delta * W3 * diag(s2) * (I @ z1^T)
    J2_core = delta * torch.matmul(W3, torch.diag(s2.flatten()))
    grad_W2_vec = torch.matmul(J2_core, torch.kron(torch.eye(d), z1.t()))
    grad_W2 = grad_W2_vec.view(d, d)
    
    # J1 using Kronecker
    J1_core = torch.matmul(J2_core, W2) * s1.t()
    grad_W1_vec = torch.matmul(J1_core, torch.kron(torch.eye(d), x.t()))
    grad_W1 = grad_W1_vec.view(d, n)
    
    return grad_W1, grad_W2, grad_W3

def backward_reverse(delta, x, z1, z2, W2, W3, a1, a2, use_relu=False):
    s2 = (a2 > 0).float() if use_relu else torch.ones_like(a2)
    s1 = (a1 > 0).float() if use_relu else torch.ones_like(a1)
    
    # grad_W3
    grad_W3 = delta * z2.t()
    
    # Upstream g2
    g2 = torch.matmul(W3.t(), delta) * s2
    grad_W2 = torch.matmul(g2, z1.t())
    
    # Upstream g1
    g1 = torch.matmul(W2.t(), g2) * s1
    grad_W1 = torch.matmul(g1, x.t())
    
    return grad_W1, grad_W2, grad_W3

def verify_parity(d=10, n=5, use_relu=True):
    x = torch.randn(n, 1)
    y = torch.randn(1, 1)
    W1 = torch.randn(d, n, requires_grad=True)
    W2 = torch.randn(d, d, requires_grad=True)
    W3 = torch.randn(1, d, requires_grad=True)
    
    # Manual forward
    a1, z1, a2, z2, z3 = forward_pass(x, W1.detach(), W2.detach(), W3.detach(), use_relu)
    delta = z3 - y
    
    # Methods
    gW1_A, gW2_A, gW3_A = backward_jacobian(delta, x, z1, z2, W2.detach(), W3.detach(), a1, a2, use_relu)
    gW1_B, gW2_B, gW3_B = backward_reverse(delta, x, z1, z2, W2.detach(), W3.detach(), a1, a2, use_relu)
    
    # Autograd
    a1_g = torch.matmul(W1, x)
    z1_g = torch.relu(a1_g) if use_relu else a1_g
    a2_g = torch.matmul(W2, z1_g)
    z2_g = torch.relu(a2_g) if use_relu else a2_g
    z3_g = torch.matmul(W3, z2_g)
    
    loss = 0.5 * torch.sum((z3_g - y)**2)
    loss.backward()
    
    assert torch.allclose(gW1_A, W1.grad, atol=1e-5)
    assert torch.allclose(gW1_B, W1.grad, atol=1e-5)
    print(f"✓ Numerical Parity Confirmed for d={d}")

def run_benchmark():
    dims = [10, 50, 100, 200, 500]
    print(f"{'d':<6} | {'Time A (s)':<12} | {'Time B (s)':<12} | {'Speedup':<10}")
    print("-" * 50)
    
    for d in dims:
        n = 10
        x = torch.randn(n, 1); y = torch.randn(1, 1)
        W1, W2, W3 = torch.randn(d, n), torch.randn(d, d), torch.randn(1, d)
        a1, z1, a2, z2, z3 = forward_pass(x, W1, W2, W3)
        delta = z3 - y
        
        start = time.perf_counter()
        for _ in range(10): backward_jacobian(delta, x, z1, z2, W2, W3, a1, a2)
        t_a = (time.perf_counter() - start) / 10
        
        start = time.perf_counter()
        for _ in range(10): backward_reverse(delta, x, z1, z2, W2, W3, a1, a2)
        t_b = (time.perf_counter() - start) / 10
        
        print(f"{d:<6} | {t_a:<12.6f} | {t_b:<12.6f} | {t_a/t_b:<10.2f}x")

if __name__ == "__main__":
    verify_parity(use_relu=False)
    verify_parity(use_relu=True)
    run_benchmark()
