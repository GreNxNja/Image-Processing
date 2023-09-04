import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

input_image = plt.imread(r"C:\Users\DELL\OneDrive\Desktop\Odin\Image Processing\levi.jpg")
if (input_image.shape[-1]==3):
    input_image = np.dot(input_image[..., :3],[0.28,0.59,0.11])

laplacian_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
filtered_image = convolve2d(input_image, laplacian_kernel,mode='same',boundary='symm')
filtered_image = np.clip(filtered_image,0,25)


plt.figure(figsize=[12,6])
plt.subplot(121)
plt.title("Original img")
plt.imshow(input_image,cmap="gray")


plt.subplot(122)
plt.title("Filtered img(laplacian)")
plt.imshow(filtered_image,cmap="gray")
plt.show()