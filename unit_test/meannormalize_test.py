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

    def test_backward_against_layernorm(self):
        
        batch_size = 4
        feature_dim = 4
        x = torch.randn(batch_size, feature_dim, requires_grad=True) - 1 # Subtract 1 to get it off center

        """Compare gradient behavior of DivisiveNormalizeFunction with LayerNorm after a linear transformation."""
        ln = self.MuNorm(feature_dim)
        linear = torch.nn.Linear(feature_dim, feature_dim, bias=False)

        # Create input with a nonzero mean
        x = x.detach().clone().requires_grad_(True)  # Independent tensor
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
        dn_out = MeanNormalizeFunction.apply(x_transformed, False)
        dn_out.backward(grad_output)  # No retain_graph needed here
        dn_weight_grad = linear.weight.grad.clone()

        # # Assert that gradients match
        # self.assertTrue(torch.allclose(ln_out, dn_out, atol=1e-4),
        #                 msg=f"Output mismatch between DivisiveNormalizeFunction and LayerNorm.\n{ln_out - dn_out}")

        # Assert that gradients match
        self.assertTrue(torch.allclose(dn_weight_grad, ln_weight_grad, atol=1e-8),
                        msg=f"Gradient mismatch between DivisiveNormalizeFunction and LayerNorm.\n{dn_weight_grad - ln_weight_grad}")