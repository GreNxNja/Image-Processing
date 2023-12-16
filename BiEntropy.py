import cv2
import numpy as np


def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    log_hist = np.log2(hist + 1e-10)
    entropy = -np.sum(hist * log_hist)
    return entropy


def bi_entropy_segmentation(image, threshold=0.1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    total_entropy = calculate_entropy(gray_image)

    binary_image = np.zeros_like(gray_image)
    for t in range(1, 255):
        foreground_pixels = gray_image[gray_image > t]
        background_pixels = gray_image[gray_image <= t]

        foreground_entropy = calculate_entropy(foreground_pixels)
        background_entropy = calculate_entropy(background_pixels)

        weight_foreground = len(foreground_pixels) / (height * width)
        weight_background = len(background_pixels) / (height * width)

        bi_entropy = weight_foreground * foreground_entropy + \
            weight_background * background_entropy

        if bi_entropy <= total_entropy * (1 - threshold):
            binary_image[gray_image > t] = 255

    return binary_image


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path)

segmented_image = bi_entropy_segmentation(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Bi-entropy Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
