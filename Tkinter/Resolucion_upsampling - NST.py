import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

path_name_input = 'C:/Users/nicor/OneDrive/Escritorio/Copiar a Drive/LR/stylized-image2-49_e_commerce.jpg'
path_name_output = 'C:/Users/nicor/OneDrive/Escritorio/Copiar a Drive/HR/maradonav2 - PH.jpg'

img = cv2.imread(path_name_input)
print(img.shape)

#plt.imshow(img)

y1 = img.shape[0]
x1 = img.shape[1]

x2 = int(input("Meter pixeles x de salida: "))
y2 = int(input("Meter pixeles y de salida: "))



#res = cv2.resize(img, None, fx=13.027, fy=11.58 , interpolation = cv2.INTER_CUBIC) #90x60
#res = cv2.resize(img, None, fx=6.513, fy=5.789 , interpolation = cv2.INTER_CUBIC)  # 45x30
res = cv2.resize(img, None, fx=x2/x1, fy=y2/y1 , interpolation = cv2.INTER_CUBIC)  # 45x30

#plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(122)
plt.imshow(res)
plt.title('Upsampled Image')

plt.show()

cv2.imwrite(path_name_output, res) 