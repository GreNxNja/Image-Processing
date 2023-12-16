import cv2
import numpy as np


def otsu_thresholding(image_path):
    # Read the image
    input_image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresholded_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return input_image, thresholded_image


def main():
    # Replace 'path/to/your/image.jpg' with the actual path to your image file
    image_path = r'levi.jpg'

    original_image, otsu_result = otsu_thresholding(image_path)

    # Display the original and thresholded images
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Otsu Thresholding Result', otsu_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
