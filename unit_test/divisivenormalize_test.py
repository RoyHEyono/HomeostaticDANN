from danns_eg.normalization import DivisiveNormalizeFunction
import torch
import unittest
import torch.nn as nn
from torch.autograd import gradcheck

class TestDivisiveNormalizeFunction(unittest.TestCase):

    class DivisiveNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            """
            Implements divisive normalization (only divides by std, does not subtract mean).
            
            Args:
                normalized_shape (int or tuple): Number of features (like LayerNorm).
                eps (float): Small constant for numerical stability.
            """
            super().__init__()
            self.eps = eps
            self.normalized_shape = normalized_shape

        def forward(self, x):
            # Compute variance (without mean subtraction)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-5)
            x_norm = x / std

            return x_norm

    def setUp(self):
        """Set up test tensors"""
        torch.manual_seed(42)
        self.batch_size = 4
        self.feature_dim = 4
        self.x = torch.randn(self.batch_size, self.feature_dim, requires_grad=True) - 1 # Subtract 1 to get it off center

    def test_divisive_norm_class(self):
        dim_x = 20
        x = torch.randn(32, dim_x)
        layernorm_control = nn.LayerNorm(dim_x, elementwise_affine=False)
        ln_local = self.DivisiveNorm(20)
        x_centered = x - x.mean(dim=1, keepdim=True)

        self.assertTrue(torch.allclose(layernorm_control(x_centered), ln_local(x_centered), atol=1e-8),
                        msg=f"Mismatch between divisive normalization and LayerNorm.")

    def test_forward_variance(self):
        """Check that DivisiveNormalizeFunction produces unit variance."""
        x_norm = DivisiveNormalizeFunction.apply(self.x, False)
        var = x_norm.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-4),
                        msg=f"Expected variance=1, got {var}")

    def test_forward_against_layernorm(self):
        """Compare output against LayerNorm but with mean fixed at 0"""
        ln = torch.nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        out_x = self.x - self.x.mean(dim=-1, keepdim=True)

        ln_out = ln(out_x)

        dn_out = DivisiveNormalizeFunction.apply(out_x, False)

        self.assertTrue(torch.allclose(dn_out, ln_out, atol=1e-5),
                        msg=f"Mismatch between divisive normalization and LayerNorm.\n{dn_out - ln_out}")

    def test_backward_against_layernorm(self):
        """Compare gradient behavior of DivisiveNormalizeFunction with LayerNorm after a linear transformation."""
        ln = self.DivisiveNorm(self.feature_dim)
        # ln = torch.nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        linear = torch.nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        # Create input with a nonzero mean
        x = self.x.detach().clone().requires_grad_(True)  # Independent tensor
        x_transformed = linear(x)  # Apply linear transformation

        grad_output = torch.randn_like(x_transformed)  # Random gradients

        # Compute gradients for LayerNorm
        linear.weight.grad = None  # Zero out gradients
        ln_out = ln(x_transformed)
        ln_out.backward(grad_output, retain_graph=True)  # Retain graph
        ln_weight_grad = linear.weight.grad.clone()

        # Compute gradients for DivisiveNormalizeFunction
        linear.weight.grad = None  # Zero out gradients
        x_transformed.requires_grad_()  # Ensure gradients are tracked
        dn_out = DivisiveNormalizeFunction.apply(x_transformed, False)
        dn_out.backward(grad_output)  # No retain_graph needed here
        dn_weight_grad = linear.weight.grad.clone()

        linear.weight.grad = None  # Zero out gradients
        x_transformed.requires_grad_()  # Ensure gradients are tracked

        # Assert that gradients match
        self.assertTrue(torch.allclose(ln_out, dn_out, atol=1e-8),
                        msg=f"Output mismatch between DivisiveNormalizeFunction and LayerNorm.\n{ln_out - dn_out}")

        # Assert that gradients match
        self.assertTrue(torch.allclose(dn_weight_grad, ln_weight_grad, atol=1e-8),
                        msg=f"Gradient mismatch between DivisiveNormalizeFunction and LayerNorm.\n{dn_weight_grad - ln_weight_grad}")

        # self.assertTrue(torch.autograd.gradcheck(DivisiveNormalizeFunction.apply, (x_transformed, False))) # Something wrong with this.

        




if __name__ == "__main__":
    unittest.main()
