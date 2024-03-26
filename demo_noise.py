import numpy as np

# Original image
original_image = np.random.rand(100, 100)

# Additive Gaussian noise with mean=0 and std=0.1
noise = np.random.normal(loc=0.5, scale=0.1, size=(100, 100))

# Add noise to the original image
noisy_image = original_image + noise

# Calculate mean and variance of original and noisy images
original_mean = np.mean(original_image)
noisy_mean = np.mean(noisy_image)

original_variance = np.var(original_image)
noisy_variance = np.var(noisy_image)

print("Original Mean:", original_mean)
print("Noisy Mean:", noisy_mean)

print("Original Variance:", original_variance)
print("Noisy Variance:", noisy_variance)
