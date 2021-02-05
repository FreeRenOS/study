import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
os.listdir("test_images/");
image = mpimg.imread('test_images/solidWhiteRight.jpg');
print('This image is:', type(image), 'with dimensions:', image.shape);

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
def draw_lines(img, lines, color=[0, 255, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # return line_img
    return lines


def hough_lines_img(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 1
# fontColor = (255, 255, 0)
# lineType = 3
def drawLine(img, x, y, flag, color=[255, 0, 0], thickness=20):
    if (len(x) == 0):
        return

    lineParameters = np.polyfit(x, y, 1)

    m = lineParameters[0]
    b = lineParameters[1]

    maxY = img.shape[0]
    maxX = img.shape[1]
    y1 = maxY
    x1 = int((y1 - b) / m)
    y2 = int((maxY / 2)) + 60
    x2 = int((y2 - b) / m)
    if (flag == 0):
        # cv2.putText(img, str(m), (50, 50), font, fontScale, fontColor, lineType)
        cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 4)
    elif (flag == 1):
        cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 4)
    else:
        cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], 4)


ht_param = [1, np.pi / 180, 25, 20, 30]
color_th = [200, 200, 10]


def enhanced_yellow(in_img):
    color_select = np.copy(in_img)
    rgb_threshold = [color_th[0], color_th[1], color_th[2]]
    color_thresholds = (in_img[:, :, 0] < rgb_threshold[0]) | \
                       (in_img[:, :, 1] < rgb_threshold[1]) | \
                       (in_img[:, :, 2] < rgb_threshold[2])
    color_select[color_thresholds] = [0, 0, 0]
    result = grayscale(color_select)
    return result


def find_lines_seg(in_image):
    height, width = in_image.shape[:2]
    gray_img = grayscale(in_image)
    smooth_img = gaussian_blur(gray_img, 5)
    edge_img = canny(smooth_img, 50, 150)
    mask = np.zeros_like(gray_img)
    ignore_mask_color = 255
    vertices = np.array([[(50, height), (width / 2 - 45, height / 2 + 60),
                          (width / 2 + 45, height / 2 + 60), (width - 50, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    ret, th_img = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img0 = cv2.bitwise_and(th_img, mask)
    img1 = cv2.bitwise_and(img0, edge_img)
    lines = hough_lines_img(img1, ht_param[0], ht_param[1], ht_param[2], ht_param[3], ht_param[4])

    return lines


def find_lines(in_image):
    height, width = in_image.shape[:2]
    gray_img = grayscale(in_image)
    smooth_img = gaussian_blur(gray_img, 5)
    edge_img = canny(smooth_img, 50, 150)
    mask = np.zeros_like(gray_img)
    ignore_mask_color = 255
    vertices = np.array([[(50, height), (width / 2 - 45, height / 2 + 60),
                          (width / 2 + 45, height / 2 + 60), (width - 50, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # ret, th_img = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, th_img = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img0 = cv2.bitwise_and(th_img, mask)
    img1 = cv2.bitwise_and(img0, edge_img)
    lines = hough_lines(img1, ht_param[0], ht_param[1], ht_param[2], ht_param[3], ht_param[4])

    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y1 - y2) / (x1 - x2)
            m_deg = m * 180 / np.pi
            if (-m_deg > 30) and (-m_deg < 60):
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
            elif (m_deg > 30) and (m_deg < 60):
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)
    return (leftPointsX, leftPointsY, rightPointsX, rightPointsY)

def find_lines_challenge(in_image):
    height, width = in_image.shape[:2]
    gray_y_img = enhanced_yellow(in_image)
    gray_img = grayscale(in_image)
    smooth_img = gaussian_blur(gray_img, 5)
    edge_img = canny(smooth_img, 50, 150)
    mask = np.zeros_like(gray_img)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(50, height), (width / 2 - 45, height / 2 + 60),
                          (width / 2 + 45, height / 2 + 60), (width - 50, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # ret, th_img = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img0 = cv2.bitwise_or(gray_y_img, edge_img)
    img1 = cv2.bitwise_and(img0, mask)
    lines = hough_lines(img1, ht_param[0], ht_param[1], ht_param[2], ht_param[3], ht_param[4])

    leftPointsX = []
    leftPointsY = []
    rightPointsX = []
    rightPointsY = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y1 - y2) / (x1 - x2)
            m_deg = m * 180 / np.pi
            if (-m_deg > 30) and (-m_deg < 60):
                leftPointsX.append(x1)
                leftPointsY.append(y1)
                leftPointsX.append(x2)
                leftPointsY.append(y2)
            elif (m_deg > 30) and (m_deg < 60):
                rightPointsX.append(x1)
                rightPointsY.append(y1)
                rightPointsX.append(x2)
                rightPointsY.append(y2)
    return (leftPointsX, leftPointsY, rightPointsX, rightPointsY)


ht_param1 = [1, np.pi / 180, 25, 20, 30]
test = mpimg.imread('test_images/solidYellowLeft.jpg')
color_select = np.copy(test)
height, width = test.shape[:2]
red_threshold = 200
green_threshold = 200
blue_threshold = 10
rgb_threshold = [red_threshold, green_threshold, blue_threshold]
color_thresholds = (test[:, :, 0] < rgb_threshold[0]) | \
                   (test[:, :, 1] < rgb_threshold[1]) | \
                   (test[:, :, 2] < rgb_threshold[2])

color_select[color_thresholds] = [0, 0, 0]
gray_img0 = grayscale(color_select);
gray_img = grayscale(test)
smooth_img = gaussian_blur(gray_img, 5)
edge_img = canny(smooth_img, 50, 150)
mask = np.zeros_like(gray_img)
ignore_mask_color = 255
imshape = test.shape
vertices = np.array(
    [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
# ret, th_img = cv2.threshold(smooth_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# img0 = cv2.bitwise_and(edge_img, mask)
img0 = cv2.bitwise_or(edge_img, gray_img0)
img1 = cv2.bitwise_and(img0, mask)
lines = hough_lines(img1, ht_param1[0], ht_param1[1], ht_param1[2], ht_param1[3], ht_param1[4])

leftPointsX = []
leftPointsY = []
rightPointsX = []
rightPointsY = []

for line in lines:
    for x1, y1, x2, y2 in line:
        m = (y1 - y2) / (x1 - x2)
        m_deg = m * 180 / np.pi
        if (-m_deg > 30) and (-m_deg < 60):
            leftPointsX.append(x1)
            leftPointsY.append(y1)
            leftPointsX.append(x2)
            leftPointsY.append(y2)
        elif (m_deg > 30) and (m_deg < 60):
            rightPointsX.append(x1)
            rightPointsY.append(y1)
            rightPointsX.append(x2)
            rightPointsY.append(y2)

# print(len(leftPointsX))
drawLine(test, leftPointsX, leftPointsY, 0)
drawLine(test, rightPointsX, rightPointsY, 1)
# line_edge = cv2.addWeighted(test, 0.8, lines, 1, 0)
# plt.imshow(lines)
#plt.imshow(test);
#plt.show();

from moviepy.editor import VideoFileClip

def process_image_seg(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    line_edge = cv2.addWeighted(image, 0.8, find_lines_seg(image), 1, 0)

    return line_edge

white_output = 'test_videos_output/solidWhiteRight_seg.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image_seg) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


