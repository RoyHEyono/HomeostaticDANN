from danns_eg import eidensenet as einet
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


class Test_EIDenseNet(unittest.TestCase):

    def test_forward_pass_output_shape(self):
        # Initialize the model for testing
        model = EIDenseNet(input_size=10, hidden_size=20, output_size=5, num_layers=3, nonlinearity=1, detachnorm=0)
        x = torch.randn(32, 10)  # Batch of 32 samples, each of size 10
        output = model(x)
        # Test that the output shape matches the expected dimensions
        self.assertEqual(output.shape, (32, 5), f"Expected output shape (32, 5), but got {output.shape}")

    def test_first_layer(self):
        # Initialize the model for testing
        model = EIDenseNet(input_size=10, hidden_size=20, output_size=5, num_layers=3, nonlinearity=1, detachnorm=0)
        x = torch.randn(32, 10)  # Batch of 32 samples, each of size 10
        output = model.fc0(x)
        # Test that the first layer is correctly initialized and has the expected output shape
        self.assertEqual(output.shape, (32, 20), f"Expected output shape (32, 20), but got {output.shape}")

    def test_relu_activation(self):
        # Initialize the model for testing
        model = EIDenseNet(input_size=10, hidden_size=20, output_size=5, num_layers=3, nonlinearity=1, detachnorm=0)
        x = torch.randn(32, 20)  # Random input for ReLU activation
        output = model.relu(x)
        # Test that ReLU is applied correctly
        self.assertTrue(torch.all(output >= 0), "ReLU did not activate correctly")

    def test_mean_normalization_layer(self):

        # Example configuration dictionary p
        p = {
            'model': {
                'hidden_layer_width': 20,
                'normtype': 1,  # Assume 1 means ReLU, or any other norm type
                'normtype_detach': 0
            },
            'exp': {
                'use_wandb': False
            }
        }

        p = dict_to_object(p)

        # Initialize the model for testing with nonlinearity
        model = einet.net(p)
        
        x = torch.randn(32, 20)
        normalized_output = model.ln(x)
        # Test that MeanNormalize works if nonlinearity is enabled
        self.assertEqual(normalized_output.shape, (32, 20), "Mean normalization output shape is incorrect")
        self.assertAlmostEqual(normalized_output.mean().item(), 0, delta=1e-5)

    def test_no_nonlinearity(self):
        # Initialize the model for testing without nonlinearity
        model = EIDenseNet(input_size=10, hidden_size=20, output_size=5, num_layers=3, nonlinearity=0, detachnorm=0)
        x = torch.randn(32, 10)
        output = model(x)
        # Test that the network works without nonlinearity
        self.assertEqual(output.shape, (32, 5), f"Expected output shape (32, 5), but got {output.shape}")

    def test_hooks_registration(self):
        # Initialize the model for testing
        model = EIDenseNet(input_size=10, hidden_size=20, output_size=5, num_layers=3, nonlinearity=1, detachnorm=0)
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
            }
        }

        p = dict_to_object(p)

        # Initialize the model for testing with nonlinearity
        model = einet.net(p)
        print(model.ln.no_backward)
        self.assertEqual(model.ln.no_backward, 1)

    # TODO: Figure this out
    # def test_hooks_removal(self):
    #     # Initialize the model for testing
    #      # Example configuration dictionary p
    #     p = {
    #         'model': {
    #             'hidden_layer_width': 20,
    #             'normtype': 1,  # Assume 1 means ReLU, or any other norm type
    #             'normtype_detach': 0
    #         },
    #         'exp': {
    #             'use_wandb': False
    #         }
    #     }

    #     p = dict_to_object(p)

    #     # Initialize the model for testing with nonlinearity
    #     model = enet.net(p)
    #     model.register_hooks()
    #     model.remove_hooks()
    #     for i in range(model.num_layers):
    #         hook_attr = getattr(model, f'fc{i}_hook', None)
    #         # Test that hooks are removed correctly
    #         self.assertIsNone(hook_attr, f"Hook for layer {i} was not removed correctly")

if __name__ == "__main__":
    unittest.main()