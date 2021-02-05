import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('test_images/solidYellowCurve.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

kernel_size = 3
smooth_img = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
#plt.imshow(smooth_img, 'gray')
#plt.show()

low_th =  50
high_th = 150
edge_img = cv2.Canny(smooth_img, low_th, high_th)
#plt.imshow(edge_img, 'gray')
#plt.show()

mask = np.zeros_like(edge_img)
ignore_mask_color = 255

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edge_img, mask)

plt.imshow(masked_edges)
plt.show()

rho = 1
theta = np.pi/180
threshold = 50
min_line_length = 30
max_line_gap = 4
line_image = np.copy(image)*0

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()