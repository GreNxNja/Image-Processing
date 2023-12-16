import cv2
import numpy as np


def global_thresholding(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the global mean intensity
    global_mean = np.mean(gray_image)

    # Apply global thresholding
    _, global_thresholded = cv2.threshold(
        gray_image, global_mean, 255, cv2.THRESH_BINARY)

    return global_thresholded


def local_thresholding(image, block_size=11, c=2):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply adaptive thresholding using the mean of the neighborhood
    local_thresholded = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )

    return local_thresholded


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path)

global_thresholded = global_thresholding(input_image)
local_thresholded = local_thresholding(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Global Thresholding Result', global_thresholded)
cv2.imshow('Local Thresholding Result', local_thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
