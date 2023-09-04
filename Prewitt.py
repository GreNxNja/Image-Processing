import numpy as np
from matplotlib import pyplot as plt

def sobelOperator(img):
    container =np.copy(img)
    size = container.shape
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            gx=(img[i-1][j-1] + img[i][j-1] + img[i+1][j-1]) - (img[i-1][j+1] + img[i][j+1] + img[i+1][j+1])
            gy=(img[i-1][j-1] + img[i-1][j] + img[i-1][j+1]) - (img[i+1][j-1] + img[i+1][j] + img[i+1][j+1])
            container[i][j]=min(255,np.sqrt(gx*2 + gy*2))
    return container

def pr(img):
    container =np.copy(img)
    size = container.shape
    for i in range(1, size[0]-1):
        for j in range(1, size[1]-1):
            gx=(img[i-1][j-1] + img[i][j-1] + img[i+1][j-1]) - (img[i-1][j+1] + img[i][j+1] + img[i+1][j+1])
            gy=(img[i-1][j-1] + img[i-1][j] + img[i-1][j+1]) - (img[i+1][j-1] + img[i+1][j] + img[i+1][j+1])
            container[i][j]=min(255,np.sqrt(gx*2 + gy*2))
    return container
img=plt.imread(r'C:\Users\DELL\OneDrive\Desktop\Odin\Image Processing\levi.jpg')
gray_img = np.dot(img[...,:3],[0.28,0.59,0.11])
sobelimg= sobelOperator(gray_img)
plt.imshow(sobelimg, cmap='gray')
plt.show()

sobelimg= pr(gray_img)
plt.imshow(sobelimg, cmap='gray')
plt.show()