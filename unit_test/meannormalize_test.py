from danns_eg.normalization import MeanNormalizeFunction, MeanNormalize
from danns_eg.eidensenet import EIDenseNet
import unittest
import torch
import numpy as np
from torch import nn

def dict_to_object(data):
    if isinstance(data, dict):
        return type('DynamicObject', (object,), {k: dict_to_object(v) for k, v in data.items()})()
    elif isinstance(data, list):
        return [dict_to_object(item) for item in data]
    else:
        return data

class TestMeanNormalizeFunction(unittest.TestCase):

    class MuNorm(nn.Module):
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
            mu = x.mean(dim=-1, keepdim=True)
            
            # Normalize without mean subtraction
            x_norm = x - mu

            return x_norm

    def test_forward(self):
        """Test whether the forward pass correctly normalizes the input."""
        torch.manual_seed(42)
        self.x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        mean = self.x.mean(dim=-1, keepdim=True)
        expected_output = self.x - mean
        output = MeanNormalizeFunction.apply(self.x, False)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_gradient_no_backward_false(self):
        """Test gradient when no_backward=False (gradients should be mean-subtracted)."""
        torch.manual_seed(42)
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        model = MeanNormalize(no_backward=False)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be mean-subtracted
        expected_grad = torch.ones_like(x) - torch.ones_like(x).mean(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(x.grad, expected_grad, atol=1e-6))

    def test_gradient_no_backward_true(self):
        """Test gradient when no_backward=True (gradients should be unchanged)."""
        torch.manual_seed(42)
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        model = MeanNormalize(no_backward=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be exactly ones (since loss is sum)
        expected_grad = torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected_grad, atol=1e-6))

    def test_gradcheck(self):
        """Test whether the function passes PyTorch's gradcheck."""
        torch.manual_seed(42)
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(MeanNormalizeFunction.apply, (x, False)))

    def test_no_forward_flag(self):
        # Create a tensor with known values
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

        # Test when no_forward=False (normal forward pass)
        normalize_fn = MeanNormalize(no_forward=False)
        output = normalize_fn(x)
        
        # Compute expected output manually (mean subtraction)
        expected_output = x - x.mean(dim=-1, keepdim=True)
        
        # Assert that the output is close to the expected output
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4), msg="Forward pass didn't normalize correctly")
        
        # Test when no_forward=True (input should not be modified)
        normalize_fn_no_forward = MeanNormalize(no_forward=True)
        output_no_forward = normalize_fn_no_forward(x)
        
        # Assert that the output is exactly the same as the input
        self.assertTrue(torch.equal(output_no_forward, x), msg="When no_forward is True, the input should not be modified")

    class LayerNormNet(nn.Module):
            """A simple network that applies a linear transformation followed by Layer Norm."""
            def __init__(self, feature_dim, mu_norm_func, use_custom_norm=False, dtype=torch.float64):
                super().__init__()
                self.linear = nn.Linear(feature_dim, feature_dim, bias=False, dtype=dtype)  # ✅ Init directly in float64
                self.norm = (
                    MeanNormalize(no_backward=False, no_forward=False).to(dtype=dtype)
                    if use_custom_norm else mu_norm_func  # ✅ Directly in float64
                )

            def forward(self, x):
                return self.norm(self.linear(x))

    def test_backward_against_layernorm(self):
        """Compare gradients of LayerNormalizeFunction vs. PyTorch's LayerNorm using autograd checks."""
        feature_dim, batch_size = 10, 5
        mu_norm_func = self.MuNorm(feature_dim)

        # ✅ Ensure input is structured and meaningful
        x = torch.arange(batch_size * feature_dim, dtype=torch.float64).view(batch_size, feature_dim)
        x.requires_grad_(True)

        # ✅ Create first model (PyTorch LayerNorm)
        model_torch = self.LayerNormNet(feature_dim, mu_norm_func, use_custom_norm=False, dtype=torch.float64)

        # ✅ Create second model (Custom LayerNorm) with **identical weights**
        model_custom = self.LayerNormNet(feature_dim, mu_norm_func, use_custom_norm=True, dtype=torch.float64)
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