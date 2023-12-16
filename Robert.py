import cv2
import numpy as np


def roberts_operator(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Roberts Cross kernels
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Apply Roberts Cross operator
    gradient_x = cv2.filter2D(gray_image, cv2.CV_64F, kernel_x)
    gradient_y = cv2.filter2D(gray_image, cv2.CV_64F, kernel_y)

    # Magnitude of the gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the magnitude to the range [0, 255]
    normalized_magnitude = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(normalized_magnitude)


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path)

roberts_result = roberts_operator(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Roberts Operator Result', roberts_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
