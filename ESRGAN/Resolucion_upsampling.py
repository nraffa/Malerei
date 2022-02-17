import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread('C:/Users/nicor/OneDrive/Documentos/Python/ESRGAN/Lab/Sudeste by Renoir Super Resolution.jpg')
#plt.imshow(img)



res = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(122)
plt.imshow(res)
plt.title('Upsampled Image')

plt.show()

cv2.imwrite('C:/Users/nicor/OneDrive/Documentos/Python/ESRGAN/Lab/Sampling/Sudeste by Renoir Super Resolution (UpSampling4).jpg', res) 