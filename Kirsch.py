import cv2
import numpy as np


def kirsch_operator(image):
    # Define Kirsch masks for eight different orientations
    masks = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ]

    # Apply each mask to the image and get the maximum response
    responses = [cv2.filter2D(image, cv2.CV_64F, mask) for mask in masks]
    gradient_magnitude = np.max(np.stack(responses, axis=-1), axis=-1)

    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(normalized_gradient)


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kirsch_result = kirsch_operator(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Kirsch Operator Result', kirsch_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
