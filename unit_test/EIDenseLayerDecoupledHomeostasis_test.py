import unittest
import torch
import numpy as np
from danns_eg.homeostaticdense import EiDenseLayerDecoupledHomeostatic
from danns_eg.dense import EiDenseLayer, EiDenseLayerMeanHomeostatic
import matplotlib.pyplot as plt

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
        z_d = torch.sqrt(z_d_squared+1e-5)

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

    def test_inhibitory_weights_gradient_updated_in_forward(self):
        """Ensure inhibitory weights receive gradients after forward pass"""
        self.layer.zero_grad()  # Clear any previous gradients

        # Run forward pass
        _ = self.layer(self.x)

        # Check if gradients exist after forward
        self.assertIsNotNone(self.layer.Wix.grad)
        self.assertFalse(self.layer.Wei.requires_grad)
        self.assertIsNotNone(self.layer.Bix.grad)
        self.assertFalse(self.layer.Bei.requires_grad)
        self.assertIsNone(self.layer.Wex.grad)
        self.assertIsNone(self.layer.b.grad)

    # TODO: test if gradients being normalized correctly
    def test_gradient_norm(self):
        pass

    def test_excitatory_weights_gradient_updated_in_backward(self):
        """Ensure Wex only receives gradients during backward, not forward"""
        self.layer.zero_grad()  # Clear gradients

        # Run forward pass
        output = self.layer(self.x)

        # Capture Wix and Wei before forward
        Wix_before = self.layer.Wix.grad.clone().detach()
        # Wei_before = self.layer.Wei.grad.clone().detach()
        Bix_before = self.layer.Bix.grad.clone().detach()
        # Bei_before = self.layer.Bei.grad.clone().detach()

        # Ensure Wex has NO gradients yet (should only update in backward)
        self.assertIsNone(self.layer.Wex.grad)
        self.assertIsNone(self.layer.b.grad)

        # Compute loss and backprop
        loss = output.sum()  # Dummy loss
        loss.backward()

        # Now Wex should have gradients
        self.assertIsNotNone(self.layer.Wex.grad)
        self.assertIsNotNone(self.layer.b.grad)
        # Ensure inhibitory weights have not changed
        self.assertTrue(torch.equal(Wix_before, self.layer.Wix.grad))
        # self.assertTrue(torch.equal(Wei_before, self.layer.Wei.grad))
        self.assertTrue(torch.equal(Bix_before, self.layer.Bix.grad))
        # self.assertTrue(torch.equal(Bei_before, self.layer.Bei.grad))