import unittest
import torch
import numpy as np
from danns_eg.homeostaticdense import EiDenseLayerDecoupledHomeostatic
from danns_eg.dense import EiDenseLayer, EiDenseLayerMeanHomeostatic

class TestEiDenseLayerDecoupledHomeostasis(unittest.TestCase):

    class DummyGradScaler:
            def scale(self, loss):
                return loss  # Returns the loss unchanged
            
            def step(self, optimizer):
                optimizer.step()  # Just calls optimizer.step() without scaling
            
            def update(self):
                pass  # No-op

            def unscale_(self, optimizer):
                pass  # No-op
    
    def setUp(self):
        torch.manual_seed(42)  # Set seed for reproducibility
        self.n_input = 784
        self.ne = 500
        self.ni = 2
        self.batch_size = 4
        
        self.layer = EiDenseLayerDecoupledHomeostatic(
            n_input=self.n_input, 
            ne=self.ne, 
            ni=self.ni, 
            lambda_homeo=0.1,
            scaler=self.DummyGradScaler(),
            init_weights_kwargs={"ex_distribution": "lognormal"}
        )
        self.ei_layer = EiDenseLayerMeanHomeostatic(n_input=self.n_input, ne=self.ne, ni=self.ni, scaler=self.DummyGradScaler())
        self.x = torch.randn(self.batch_size, self.n_input)

    def test_forward_output_numerically(self):
        # Compute excitatory input by projecting x onto Wex
        hex = torch.matmul(self.x, self.layer.Wex.T)
        
        # Compute inhibitory input, but detach x to prevent gradients from flowing back to x
        hi = torch.matmul(self.x, self.layer.Wix.T)
        
        # Compute inhibitory output
        hi = torch.matmul(hi, self.layer.Wei.T)

        if self.layer.use_bias: 
            hex = hex + self.layer.b.T

        # Add divisive variance here...
        z_d_squared = torch.matmul(torch.matmul(self.x, self.layer.Bix.T)**2, self.layer.Bei.T)
        z_d = torch.sqrt(z_d_squared)

        output = (hex - hi) / z_d
        
        self.assertTrue(torch.equal(output, self.layer(self.x)), msg="Numerical test for forward")

    def test_init_moments(self):
        output = self.layer(self.x)

        var = output.var(dim=-1, unbiased=False)
        mu = output.mean(dim=-1)

        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-6),
                        msg=f"Expected variance=1, got {var}")
        self.assertTrue(torch.allclose(mu, torch.zeros_like(mu), atol=1e-6),
                        msg=f"Expected mu=0, got {mu}")

    def test_forward_against_layernorm_at_init(self):
        """Compare output against LayerNorm but with mean fixed at 0"""
        ln = torch.nn.LayerNorm(self.ne, elementwise_affine=False)

        # Compute excitatory input by projecting x onto Wex
        hex = torch.matmul(self.x, self.layer.Wex.T)

        excitatory_output = ln(hex)
        decoupled_homeostatic_output = self.layer(self.x)

        self.assertTrue(torch.allclose(excitatory_output, decoupled_homeostatic_output, atol=1e-6),
                        msg=f"Initialization doesn't match ln on excitatory output")


    def test_forward_output_shape(self):
        """Test that the forward method returns the correct output shape."""
        self.x = torch.randn(self.batch_size, self.n_input)
        output = self.layer.forward(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.ne))
    
    def test_weights_initialized_correctly(self):
        """Test that weights are initialized correctly."""

        self.assertEqual(self.layer.Wex.shape, (self.ne, self.n_input))
        self.assertEqual(self.layer.Wix.shape, (self.ni, self.n_input))
        self.assertEqual(self.layer.Wei.shape, (self.ne, self.ni))