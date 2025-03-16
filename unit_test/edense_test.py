from danns_eg.dense import EiDenseLayer, EDenseLayer
import unittest
import torch
import numpy as np
from torch import nn

class Test_EDenseLayer(unittest.TestCase):

    def test_edense_initialization(self):
        layer = EDenseLayer(n_input=10, ne=5)
        assert layer.Wex.shape == (5, 10), "Weight matrix shape is incorrect"
        assert layer.b.shape == (5, 1), "Bias shape is incorrect"

    def test_edense_forward(self):
        layer = EDenseLayer(n_input=10, ne=5)
        x = torch.randn(2, 10)  # Batch of 2
        output = layer(x)
        assert output.shape == (2, 5), "Output shape is incorrect"

    
    def test_edenselayer_forward_numerically(self):
        # Define small input and network
        ne, n_input = 3, 2
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape: (3,2)
        
        # Create layer with fixed weights and bias
        layer = EDenseLayer(n_input=n_input, ne=ne, use_bias=True)

        # Compute expected output manually
        expected_output = torch.matmul(x, layer.Wex.T) + layer.bias.T  # (3,3) + (1,3) -> (3,3)

        # Get actual output
        output = layer(x)

        # Check numerical equality
        assert torch.allclose(output, expected_output, atol=1e-6), f"Expected {expected_output}, but got {output}"

    def test_edenselayer_gradients_exact(self):
        """Test gradient computation with two neurons, one input, and bias."""
        # Input tensor: shape (1,1)
        x = torch.tensor([[2.0]], requires_grad=True)  # Batch size = 1, input dim = 1

        # Define EDenseLayer with 2 excitatory neurons and fixed weights & bias
        layer = EDenseLayer(n_input=1, ne=2, use_bias=True)
        layer.Wex = nn.Parameter(torch.tensor([[3.0], [4.0]]))  # Shape: (2,1)
        layer.bias = nn.Parameter(torch.tensor([[1.0], [2.0]]))  # Shape: (2,1)

        # Forward pass
        output = layer(x)  # Expected output: [[6+1], [8+2]] = [[7], [10]]

        # Compute loss
        loss = output.sum()  # L = 7 + 10 = 17
        loss.backward()

        # Expected gradients
        expected_grad_Wex = torch.tensor([[2.0], [2.0]])  # dL/dWex = x (broadcasted)
        expected_grad_bias = torch.tensor([[1.0], [1.0]])  # dL/db = 1 (broadcasted)
        expected_grad_x = torch.tensor([[3.0 + 4.0]])  # dL/dx = sum of Wex rows

        # Compare gradients
        assert torch.allclose(layer.Wex.grad, expected_grad_Wex, atol=1e-6), \
            f"Expected Wex grad {expected_grad_Wex}, got {layer.Wex.grad}"

        assert torch.allclose(layer.bias.grad, expected_grad_bias, atol=1e-6), \
            f"Expected bias grad {expected_grad_bias}, got {layer.bias.grad}"

        assert torch.allclose(x.grad, expected_grad_x, atol=1e-6), \
            f"Expected x grad {expected_grad_x}, got {x.grad}"

    def test_edense_no_bias(self):
        layer = EDenseLayer(n_input=10, ne=5, use_bias=False)
        assert layer.b is None, "Bias should be None when use_bias is False"

    def test_edense_nonlinearity(self):
        relu = torch.nn.ReLU()
        layer = EDenseLayer(n_input=10, ne=5, nonlinearity=relu)
        x = torch.randn(2, 10)
        output = layer(x)
        assert torch.all(output >= 0), "ReLU nonlinearity not applied correctly"

    def test_edense_weight_initialization(self):
        layer1 = EDenseLayer(n_input=10, ne=5)
        layer2 = EDenseLayer(n_input=10, ne=5)
        assert not torch.equal(layer1.Wex, layer2.Wex), "Weights should not be identical across instances"

    def test_edense_zero_input(self):
        layer = EDenseLayer(n_input=10, ne=5)
        x = torch.zeros(2, 10)
        output = layer(x)
        if layer.use_bias:
            assert torch.all(output == layer.b.T), "Output should match bias when input is zero"
        else:
            assert torch.all(output == 0), "Output should be zero when input and bias are zero"

    def test_edenselayer_gradients(self):
        torch.manual_seed(0)  # For reproducibility
        n_input = 10
        ne = 5
        batch_size = 3
        x = torch.randn(batch_size, n_input, requires_grad=True)
        
        # Create layer
        layer = EDenseLayer(n_input=n_input, ne=ne)
        output = layer(x)
        
        # Define a simple loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are not None and have expected shapes
        assert layer.Wex.grad is not None, "Wex gradients should not be None"
        assert layer.Wex.grad.shape == (ne, n_input), f"Expected shape {(ne, n_input)}, got {layer.Wex.grad.shape}"
        
        if layer.use_bias:
            assert layer.b.grad is not None, "Bias gradients should not be None"
            assert layer.b.grad.shape == (ne, 1), f"Expected shape {(ne, 1)}, got {layer.b.grad.shape}"
        
        # Check that input x has gradients
        assert x.grad is not None, "Input x should have gradients"
        assert x.grad.shape == (batch_size, n_input), f"Expected shape {(batch_size, n_input)}, got {x.grad.shape}"

if __name__ == '__main__':
    unittest.main()