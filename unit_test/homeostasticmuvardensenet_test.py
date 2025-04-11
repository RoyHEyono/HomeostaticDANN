from danns_eg import homeostatic_mu_var_densenet as homeostaticnet
from danns_eg.homeostatic_mu_var_densenet import HomeostaticDenseDANN
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


class Test_HomeostasticDenseNet(unittest.TestCase):

    class DummyGradScaler:
            def scale(self, loss):
                return loss  # Returns the loss unchanged
            
            def step(self, optimizer):
                optimizer.step()  # Just calls optimizer.step() without scaling
            
            def update(self):
                pass  # No-op

            def unscale_(self, optimizer):
                pass  # No-op

    def test_forward_pass_output_shape(self):
        # Initialize the model for testing
        model = HomeostaticDenseDANN(input_size=20, hidden_size=10, output_size=5, scaler=self.DummyGradScaler(), detachnorm=0)
        x = torch.randn(32, 20)  # Batch of 32 samples, each of size 10
        output = model(x)
        # Test that the output shape matches the expected dimensions
        self.assertEqual(output.shape, (32, 5), f"Expected output shape (32, 5), but got {output.shape}")

    def test_first_layer(self):
        # Initialize the model for testing
        model = HomeostaticDenseDANN(input_size=20, hidden_size=10, output_size=5, scaler=self.DummyGradScaler(), detachnorm=0)
        x = torch.randn(32, 20)  # Batch of 32 samples, each of size 10
        output = model.fc0(x)
        # Test that the first layer is correctly initialized and has the expected output shape
        self.assertEqual(output.shape, (32, 10), f"Expected output shape (32, 10), but got {output.shape}")

    def test_relu_activation(self):
        # Initialize the model for testing
        model = HomeostaticDenseDANN(input_size=20, hidden_size=10, output_size=5, scaler=self.DummyGradScaler(), detachnorm=0)
        x = torch.randn(32, 20)  # Random input for ReLU activation
        output = model.relu(x)
        # Test that ReLU is applied correctly
        self.assertTrue(torch.all(output >= 0), "ReLU did not activate correctly")

    def test_hooks_registration(self):
        # Initialize the model for testing
        model = HomeostaticDenseDANN(input_size=20, hidden_size=10, output_size=5, scaler=self.DummyGradScaler(), detachnorm=0)
        model.register_hooks()
        for i in range(model.num_layers):
            hook_attr = getattr(model, f'fc{i}_hook', None)
            # Test that hooks are registered correctly
            self.assertIsNotNone(hook_attr, f"Hook for layer {i} is not registered correctly")

    def test_detach(self):
        # Initialize the model for testing
        # Example configuration dictionary p
        p = {
            'model': {
                'hidden_layer_width': 20,
                'normtype': 1,  # Assume 1 means ReLU, or any other norm type
                'normtype_detach': 1
            },
            'exp': {
                'use_wandb': False
            },
            'opt': {
                'lambda_homeo': 1,
                'lambda_homeo_var': 1
            }
        }

        p = dict_to_object(p)

        # Initialize the model for testing with nonlinearity
        model = homeostaticnet.net(p, self.DummyGradScaler())

        self.assertTrue(model.fc0.apply_ln_grad.no_backward)

if __name__ == "__main__":
    unittest.main()