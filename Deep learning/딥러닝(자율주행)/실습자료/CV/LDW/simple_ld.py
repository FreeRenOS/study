import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('test_images/solidWhiteRight.jpg')
color_select = np.copy(image)
height = image.shape[0]
width  = image.shape[1]
print(height, width)

r_th = 200
g_th = 200
b_th = 200
rgb_th = [r_th, g_th, b_th]

th = (image[:,:,0] < rgb_th[0])\
   | (image[:,:,1] < rgb_th[1])\
   | (image[:,:,2] < rgb_th[2])

color_select[th] = [0, 0, 0]

plt.imshow(color_select)
plt.show()

mask = np.zeros_like(color_select)
ignore_mask_color = 255
vertices = np.array([[(50, height), (width/2-45, height/2+60),
                      (width/2+45, height/2+60), (width-50, height)]], dtype=np.int32)

cv2.fillPoly(mask, vertices, ignore_mask_color)
plt.imshow(mask, 'gray')
plt.show()

img0 = cv2.bitwise_and(color_select, mask)
result = cv2.addWeighted(image, 0.4, img0, 1, 0) #result=image*0.4 + 1*img0+0.0
plt.imshow(result)
plt.show()
