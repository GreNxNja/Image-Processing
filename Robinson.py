import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

input_image = plt.imread(r"C:\Users\DELL\OneDrive\Desktop\Odin\Image Processing\levi.jpg")
if (input_image.shape[-1]==3):
    input_image = np.dot(input_image[..., :3],[0.28,0.59,0.11])

def robinson_operator(image):
    kernels=[
        np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
        np.array([[-2,-1,0],[-1,0,1],[0,1,2]]),
        np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
        np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    ]
    gradient_magnitudes= [convolve2d(image,kernel,mode='same',boundary='symm') for kernel in kernels]
    gradient_magnitude= np.max(gradient_magnitudes,axis=0)
    return gradient_magnitude

filtered_image= robinson_operator(input_image)

plt.figure(figsize=[12,6])
plt.subplot(121)
plt.title("Original img")
plt.imshow(input_image,cmap="gray")

plt.subplot(122)
plt.title("Filtered img(robinson)")
plt.imshow(filtered_image,cmap="gray")
plt.show()