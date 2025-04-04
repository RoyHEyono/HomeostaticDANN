from danns_eg.normalization import LayerNormalizeCustom, LayerNormalizeFunction
import torch
import unittest
import torch.nn as nn
from torch.autograd import gradcheck

class TestLayerNormalizeFunction(unittest.TestCase):

    def setUp(self):
        """Set up test tensors"""
        torch.manual_seed(42)
        self.batch_size = 4
        self.feature_dim = 4
        self.x = torch.randn(self.batch_size, self.feature_dim, requires_grad=True) - 1 # Subtract 1 to get it off center

    def test_forward_moments(self):
        """Check that DivisiveNormalizeFunction produces unit variance."""
        ln_custom = LayerNormalizeCustom(no_backward=False, no_forward=False)
        x_norm = ln_custom(self.x)
        var = x_norm.var(dim=-1, unbiased=False)
        mu = x_norm.mean(dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-4),
                        msg=f"Expected variance=1, got {var}")
        self.assertTrue(torch.allclose(mu, torch.zeros_like(mu), atol=1e-4),
                        msg=f"Expected mean=0, got {mu}")

    def test_forward_against_layernorm(self):
        """Compare output against LayerNorm but with mean fixed at 0"""
        ln = torch.nn.LayerNorm(self.feature_dim, elementwise_affine=False)

        ln_out = ln(self.x)

        ln_custom = LayerNormalizeCustom(no_backward=False, no_forward=False)
        dn_out = ln_custom(self.x)

        self.assertTrue(torch.allclose(dn_out, ln_out, atol=1e-5),
                        msg=f"Mismatch between divisive normalization and LayerNorm.\n{dn_out - ln_out}")

    class LayerNormNet(nn.Module):
        """A simple network that applies a linear transformation followed by Layer Norm."""
        def __init__(self, feature_dim, use_custom_norm=False, dtype=torch.float64):
            super().__init__()
            self.linear = nn.Linear(feature_dim, feature_dim, bias=False, dtype=dtype)  # ✅ Init directly in float64
            self.norm = (
                LayerNormalizeCustom(no_backward=False, no_forward=False).to(dtype=dtype)
                if use_custom_norm else nn.LayerNorm(feature_dim, elementwise_affine=False, dtype=dtype)  # ✅ Directly in float64
            )

        def forward(self, x):
            return self.norm(self.linear(x))

    def test_backward_against_layernorm(self):
        """Compare gradients of LayerNormalizeFunction vs. PyTorch's LayerNorm using autograd checks."""
        feature_dim, batch_size = 10, 5

        # ✅ Ensure input is structured and meaningful
        x = torch.arange(batch_size * feature_dim, dtype=torch.float64).view(batch_size, feature_dim)
        x.requires_grad_(True)

        # ✅ Create first model (PyTorch LayerNorm)
        model_torch = self.LayerNormNet(feature_dim, use_custom_norm=False, dtype=torch.float64)

        # ✅ Create second model (Custom LayerNorm) with **identical weights**
        model_custom = self.LayerNormNet(feature_dim, use_custom_norm=True, dtype=torch.float64)
        model_custom.linear.weight.data.copy_(model_torch.linear.weight.data)  # ✅ Copy identical weights

        # Compute scalar loss (sum ensures gradients exist)
        loss_torch = model_torch(x).sum()
        loss_custom = model_custom(x).sum()

        # Compute gradients w.r.t. x
        grad_x_torch, = torch.autograd.grad(loss_torch, x, create_graph=True)
        grad_x_custom, = torch.autograd.grad(loss_custom, x, create_graph=True)

        # ✅ Compare outputs and gradients
        self.assertTrue(torch.allclose(loss_torch, loss_custom, atol=1e-25),
                        msg=f"Output mismatch: {loss_torch.item()} vs {loss_custom.item()}")

        self.assertTrue(torch.allclose(grad_x_torch, grad_x_custom, atol=1e-8),
                        msg=f"Gradient mismatch.\n{grad_x_torch - grad_x_custom}")

        # ✅ Check parameter gradients
        model_torch.zero_grad()
        model_custom.zero_grad()
        loss_torch.backward(retain_graph=True)
        loss_custom.backward()
        self.assertTrue(torch.allclose(model_torch.linear.weight.grad, model_custom.linear.weight.grad, atol=1e-8),
                        msg="Linear weight gradients mismatch.")

        # ✅ Use gradcheck on the **custom model**, ensuring everything is float64
        x_gradcheck = x.detach().clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(model_custom, (x_gradcheck,), eps=1e-6, atol=1e-4))

        




if __name__ == "__main__":
    unittest.main()
