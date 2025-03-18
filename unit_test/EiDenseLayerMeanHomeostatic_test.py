import unittest
import torch
import numpy as np
from danns_eg.dense import EiDenseLayerMeanHomeostatic

class TestEiDenseLayerMeanHomeostatic(unittest.TestCase):

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
        self.n_input = 5
        self.ne = 3
        self.ni = 2
        self.batch_size = 4
        
        self.layer = EiDenseLayerMeanHomeostatic(
            n_input=self.n_input, 
            ne=self.ne, 
            ni=self.ni, 
            lambda_homeo=0.1,
            scaler=self.DummyGradScaler()
        )
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

        output = hex - hi
        
        self.assertTrue(torch.equal(output, self.layer(self.x)), msg="Numerical test for forward")
    
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
    
    def test_forward_does_not_crash(self):
        """Ensure that the forward pass runs without errors."""
        try:
            self.layer.forward(self.x)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
    
    def test_gradients_update_correctly(self):
        """Test that the gradients for inhibitory weights are computed correctly."""
        self.layer.zero_grad()
        output = self.layer.forward(self.x)
        loss = output.mean()
        loss.backward()
        
        self.assertIsNotNone(self.layer.Wex.grad)  # Wex should have gradients
        self.assertIsNotNone(self.layer.Wei.grad)  # Wei should have gradients
        self.assertIsNotNone(self.layer.Wix.grad)  # Wix should have gradients
    
    def test_local_loss_computation(self):
        """Ensure that the local loss is computed correctly."""
        output = self.layer.forward(self.x)
        self.assertIsInstance(self.layer.local_loss_value, float)
    
    def test_inhibitory_weights_gradient_updated_in_forward(self):
        """Ensure inhibitory weights receive gradients after forward pass"""
        self.layer.zero_grad()  # Clear any previous gradients

        # Run forward pass
        _ = self.layer(self.x)

        # Check if gradients exist after forward
        self.assertIsNotNone(self.layer.Wix.grad)
        self.assertIsNotNone(self.layer.Wei.grad)
        self.assertIsNone(self.layer.Wex.grad)
        self.assertIsNone(self.layer.b.grad)

    def test_excitatory_weights_gradient_updated_in_backward(self):
        """Ensure Wex only receives gradients during backward, not forward"""
        self.layer.zero_grad()  # Clear gradients

        # Run forward pass
        output = self.layer(self.x)

        # Capture Wix and Wei before forward
        Wix_before = self.layer.Wix.grad.clone().detach()
        Wei_before = self.layer.Wei.grad.clone().detach()

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
        self.assertTrue(torch.equal(Wei_before, self.layer.Wei.grad))

    def test_inhibitory_weights_receive_local_loss(self):
        """Ensure homeostatic loss affects inhibitory weights"""
        self.layer.zero_grad()
        _ = self.layer(self.x)

        # Ensure inhibitory weights have nonzero gradients
        self.assertTrue(torch.any(self.layer.Wix.grad != 0))
        self.assertTrue(torch.any(self.layer.Wei.grad != 0))

    def test_excitatory_weights_receive_main_loss(self):
        """Ensure excitatory weights only receive gradients from main loss"""
        self.layer.zero_grad()
        output = self.layer(self.x)

        # Dummy loss
        loss = (output ** 2).sum()
        loss.backward()

        # Ensure Wex receives nonzero gradients
        self.assertTrue(torch.any(self.layer.Wex.grad != 0))
        self.assertTrue(torch.any(self.layer.b.grad != 0))

    def test_gradient_norm_flag(self):
        # Test for gradient_norm=False
        model_no_grad_norm = EiDenseLayerMeanHomeostatic(
            n_input=10, 
            ne=5, 
            ni=0.1, 
            gradient_norm=False,
            scaler=self.DummyGradScaler()
        )

        # Create a dummy input tensor
        x = torch.randn(2, 10, requires_grad=True)

        # Perform forward pass
        output_no_grad_norm = model_no_grad_norm(x)
        
        # Perform backward pass
        output_no_grad_norm.mean().backward()
        
        # Get the gradient w.r.t Wex
        grad_wex_no_grad_norm = model_no_grad_norm.Wex.grad

        # Additionally, check that the gradient for the `Wex` is properly centered when gradient_norm=True
        mean_no_grad = grad_wex_no_grad_norm.mean()
        self.assertNotAlmostEqual(mean_no_grad.item(), 0, delta=1e-4, 
                          msg="When gradient_norm is False, the mean gradient should not be close to 0")

        
        # Test for gradient_norm=True
        model_grad_norm = EiDenseLayerMeanHomeostatic(
            n_input=10, 
            ne=5, 
            ni=0.1, 
            gradient_norm=True,
            scaler=self.DummyGradScaler()
        )
        
        # Perform forward pass
        output_grad_norm = model_grad_norm(x)
        
        # Perform backward pass
        output_grad_norm.mean().backward()
        
        # Get the gradient w.r.t Wex
        grad_wex_grad_norm = model_grad_norm.Wex.grad

        # Compare the gradients
        self.assertFalse(torch.equal(grad_wex_no_grad_norm, grad_wex_grad_norm), 
                         msg="Gradients should differ between gradient_norm=True and gradient_norm=False")
        
        # Additionally, check that the gradient for the `Wex` is properly centered when gradient_norm=True
        mean_grad = grad_wex_grad_norm.mean()
        self.assertAlmostEqual(mean_grad.item(), 0, delta=1e-4, 
                               msg="When gradient_norm is True, the mean gradient should be close to 0")
    
if __name__ == '__main__':
    unittest.main()
