import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d 

# Load the input image
input_image = plt.imread(r'C:\Users\DELL\OneDrive\Desktop\Odin\Image Processing\levi.jpg')

# Check if the image has 3 channels (RGB), and if so, convert it to grayscale
if input_image.shape[-1] == 3:
    input_image = np.dot(input_image[..., :3], [0.28, 0.59, 0.11])

# Prompt the user for filter size and sigma
filter_size = int(input("Enter filter size (odd integer): "))
sigma = float(input("Enter sigma (e.g., 1.5): "))

# Ensure filter_size is odd
if filter_size % 2 == 0:
    filter_size += 1

# Define the Gaussian kernel function
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            -((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ),
        (size, size),
    )
    return kernel / np.sum(kernel)

# Create the Gaussian filter kernel
gaussian_filter_kernel = gaussian_kernel(filter_size, sigma)

# Apply Gaussian smoothing to the input image
filtered_image = convolve2d(input_image, gaussian_filter_kernel, mode='same', boundary='symm')

# Display the original and filtered images
plt.figure(figsize=[12, 6])
plt.subplot(121)
plt.title("Original Image")
plt.imshow(input_image, cmap="gray")

plt.subplot(122)
plt.title(f"Filtered Image (Gaussian, Sigma={sigma})")
plt.imshow(filtered_image, cmap="gray")
plt.show()