# Image Processing Toolbox

This toolbox provides a set of Python scripts for various image processing tasks. Each script is designed to perform a specific operation on images.

## List of Files

- **BiEntropy.py**: Implements the Bi-Entropy thresholding technique for image segmentation.

- **Gaussian.py**: Applies Gaussian smoothing to images for noise reduction.

- **GlobLoc.py**: Computes global and local thresholding for image segmentation.

- **HistoThresh.py**: Implements histogram thresholding for image segmentation.

- **IterativeSelection.py**: Performs iterative selection of image features.

- **Kirsch.py**: Applies the Kirsch operator for edge detection.

- **Laplacian.py**: Utilizes the Laplacian operator for image sharpening.

- **Otsu.py**: Implements Otsu's method for automatic image thresholding.

- **Prewitt.py**: Applies the Prewitt operator for edge detection.

- **Robert.py**: Utilizes the Robert operator for edge detection.

- **Robinson.py**: Applies the Robinson operator for edge detection.

- **Sobel.py**: Utilizes the Sobel operator for edge detection.

## Usage

Each script is designed to be used independently. To use a script, open your terminal and navigate to the directory containing the script file. Then, run the script with the following syntax:

```bash
python ScriptName.py input_image.jpg output_image.jpg
```

Replace `ScriptName.py` with the name of the script you want to use, `input_image.jpg` with the path to your input image file, and `output_image.jpg` with the desired path for the output image.

## Dependencies

- Python 3.12
- NumPy
- OpenCV (for image I/O and basic operations)

Install the dependencies using:

```bash
pip install numpy opencv-python
```

## License

This project is licensed under the [MIT License](LICENSE).

