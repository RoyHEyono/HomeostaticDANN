from danns_eg.dense import EiDenseLayer, EDenseLayer
import unittest
import torch
import numpy as np
from torch import nn

class Test_EIDenseLayer(unittest.TestCase):

    def test_initialization(self):
        layer = EiDenseLayer(n_input=10, ne=20, ni=5)
        assert layer.Wex.shape == (20, 10)
        assert layer.Wix.shape == (5, 10)
        assert layer.Wei.shape == (20, 5)
        if layer.use_bias:
            assert layer.bias.shape == (20, 1)

    def test_weight_properties(self):
        layer = EiDenseLayer(n_input=10, ne=20, ni=5)
        W_computed = layer.W
        assert W_computed.shape == (20, 10)

    def test_forward_pass(self):
        torch.manual_seed(42)
        layer = EiDenseLayer(n_input=10, ne=20, ni=5, nonlinearity=nn.ReLU())
        x = torch.randn(4, 10)  # Batch size 4, input size 10
        output = layer(x)
        assert output.shape == (4, 20)
    
    # TODO: Test this numerically
    def test_eidenselayer_forward_numerically(self):
        """Numerically verify the forward pass of EiDenseLayer."""
    
        # Define input
        x = torch.tensor([[2.0, 3.0]])  # Shape: (1,2)

        # Define EiDenseLayer with 2 excitatory neurons, 1 inhibitory neuron, and bias
        layer = EiDenseLayer(n_input=2, ne=2, ni=1, use_bias=True)

        # Compute forward pass
        output = layer(x)  # Shape: (1,2)

        # Expected output
        effective_weight_matrix = (layer.Wex - torch.matmul(layer.Wei, layer.Wix))
        expected_output = torch.matmul(x, effective_weight_matrix.T) + layer.bias.T

        # Compare outputs
        assert torch.allclose(output, expected_output, atol=1e-6), \
            f"Expected {expected_output}, got {output}"

    def test_no_bias(self):
        layer = EiDenseLayer(n_input=10, ne=20, ni=5, use_bias=False)
        assert layer.bias is None

    def test_bias_influence(self):
        torch.manual_seed(42)
        layer_with_bias = EiDenseLayer(n_input=10, ne=20, ni=5, use_bias=True)
        layer_without_bias = EiDenseLayer(n_input=10, ne=20, ni=5, use_bias=False)
        x = torch.randn(4, 10)
        out_with_bias = layer_with_bias(x)
        out_without_bias = layer_without_bias(x)
        assert not torch.equal(out_with_bias, out_without_bias)

    def test_random_weight_initialization(self):
        layer1 = EiDenseLayer(n_input=10, ne=20, ni=5)
        layer2 = EiDenseLayer(n_input=10, ne=20, ni=5)
        assert not torch.equal(layer1.Wex, layer2.Wex)  # Ensure different initializations

    def test_nonlinearity(self):
        layer = EiDenseLayer(n_input=10, ne=20, ni=5, nonlinearity=nn.ReLU())
        x = torch.randn(4, 10)
        output = layer(x)
        assert torch.all(output >= 0)  # ReLU should not produce negative values

    def test_edge_case_no_inhibitory(self):
        layer = EiDenseLayer(n_input=10, ne=20, ni=0)
        assert layer.Wix.shape == (0, 10)
        assert layer.Wei.shape == (20, 0)
        x = torch.randn(4, 10)
        output = layer(x)
        assert output.shape == (4, 20)

    def test_eidenselayer_gradients_numerically(self):
        """Test gradient computation in EiDenseLayer with simple, predictable weights and bias."""
        # Input tensor (requires grad)
        x = torch.tensor([[2.0, 3.0]], requires_grad=True)  # Shape: (1,2)

        # Define EiDenseLayer with 2 excitatory neurons, 1 inhibitory neuron, and bias
        layer = EiDenseLayer(n_input=2, ne=2, ni=1, use_bias=True)

        # Manually set weights and bias
        layer.Wex = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))  # Shape: (2,2)
        layer.Wix = nn.Parameter(torch.tensor([[0.5, 0.5]]))  # Shape: (1,2)
        layer.Wei = nn.Parameter(torch.tensor([[1.0], [1.0]]))  # Shape: (2,1)
        layer.bias = nn.Parameter(torch.tensor([[1.0], [2.0]]))  # Shape: (2,1)

        # Compute forward pass
        output = layer(x)

        # Compute loss (sum of output)
        loss = output.sum()
        loss.backward()

        # Expected weight gradients:
        # dL/dWex = x (broadcasted)
        expected_grad_Wex = torch.tensor([[2.0, 3.0], [2.0, 3.0]])

        # dL/dWix = - sum(Wei * x) across all inhibitory neurons
        # Since Wei = [[1.0], [1.0]], the sum is over 1.0 * x per inhibitory neuron
        expected_grad_Wix = -torch.tensor([[4.0, 6.0]])  # - sum(x * Wei.T)

        # dL/dWei = - sum(Wix * x) across all inhibitory neurons
        expected_grad_Wei = -torch.tensor([[2.5], [2.5]])  # - sum(Wix * x)

        # dL/db = 1 (broadcasted)
        expected_grad_bias = torch.tensor([[1.0], [1.0]])

        # Compare gradients
        assert torch.allclose(layer.Wex.grad, expected_grad_Wex, atol=1e-6), \
            f"Expected Wex grad {expected_grad_Wex}, got {layer.Wex.grad}"

        assert torch.allclose(layer.Wix.grad, expected_grad_Wix, atol=1e-6), \
            f"Expected Wix grad {expected_grad_Wix}, got {layer.Wix.grad}"

        assert torch.allclose(layer.Wei.grad, expected_grad_Wei, atol=1e-6), \
            f"Expected Wei grad {expected_grad_Wei}, got {layer.Wei.grad}"

        assert torch.allclose(layer.bias.grad, expected_grad_bias, atol=1e-6), \
            f"Expected bias grad {expected_grad_bias}, got {layer.bias.grad}"

if __name__ == '__main__':
    unittest.main()