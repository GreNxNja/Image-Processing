import cv2
import numpy as np


def find_valley_points(hist):
    valley_points = []
    for i in range(1, len(hist)-1):
        if hist[i-1] > hist[i] < hist[i+1]:
            valley_points.append(i)
    return valley_points


def histogram_valley_thresholding(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the histogram
    hist, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

    # Find valley points in the histogram
    valley_points = find_valley_points(hist)

    # Choose the threshold as the midpoint between valley points
    thresholds = [(valley_points[i] + valley_points[i+1]) //
                  2 for i in range(0, len(valley_points)-1, 2)]

    # Apply thresholding
    segmented_image = np.zeros_like(gray_image)
    for threshold in thresholds:
        segmented_image[gray_image > threshold] = 255

    return segmented_image


# Example usage
image_path = r'levi.jpg'
input_image = cv2.imread(image_path)

segmented_image = histogram_valley_thresholding(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Histogram Valley Thresholding Result', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
