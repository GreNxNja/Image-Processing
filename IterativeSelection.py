import cv2
import numpy as np


def iterative_selection(image, initial_threshold=128, max_iterations=100, convergence_threshold=1e-3):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Initialize threshold
    threshold = initial_threshold

    for iteration in range(max_iterations):
        # Segment the image based on the current threshold
        segmented_image = np.zeros_like(gray_image)
        segmented_image[gray_image > threshold] = 255

        # Calculate mean intensity values for both regions
        foreground_mean = np.mean(gray_image[segmented_image == 255])
        background_mean = np.mean(gray_image[segmented_image == 0])

        # Update the threshold as the mean of the two regions
        new_threshold = (foreground_mean + background_mean) / 2

        # Check for convergence
        if abs(new_threshold - threshold) < convergence_threshold:
            break

        threshold = new_threshold

    # Apply the final threshold to segment the image
    _, segmented_image = cv2.threshold(
        gray_image, threshold, 255, cv2.THRESH_BINARY)

    return segmented_image


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path)

segmented_image = iterative_selection(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Iterative Selection Segmentation Result', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
