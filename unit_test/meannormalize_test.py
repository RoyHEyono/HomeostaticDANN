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