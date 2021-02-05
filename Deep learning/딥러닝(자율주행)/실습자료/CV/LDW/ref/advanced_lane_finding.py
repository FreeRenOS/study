#!/usr/bin/env python
# coding: utf-8

# In[93]:


import os
import pickle

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


def imread_rgb(path):
    return cv2.imread(imgpath)[:, :, ::-1]


# In[94]:


class Camera(object):
    
    def __init__(self):
        self.mtx = None
        self.dist = None
        src = np.float32([(220, 720), 
                          (570, 470), 
                          (720, 470),
                          (1110, 720)])
        dst = np.float32([(370, 720), 
                          (370, 0), 
                          (870, 0),
                          (870, 720)])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def calibrate(self):
        try:
            with open('camera.p', 'rb') as f:
                dist_pickle = pickle.load(f)
            self.mtx = dist_pickle['mtx']
            self.dist = dist_pickle['dist']
            print('Loaded calibration coefficients')
            return
        except:
            print('Calibrating camera')

        nx, ny = 9, 6
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)
        objpoints, imgpoints = [], []

        for imgpath in glob('./camera_cal/*.jpg'):
            img = cv2.imread(imgpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret is True:
                imgpoints.append(corners)
                objpoints.append(objp)

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        with open('camera.p', 'wb') as f:
            pickle.dump({
                'mtx': self.mtx,
                'dist': self.dist
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def undistort(self, image):
        if self.mtx is None:
            self.calibrate()
        
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
    def warp(self, image):
        return cv2.warpPerspective(image, self.M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    def unwarp(self, image):
        return cv2.warpPerspective(image, self.M_inv, image.shape[1::-1], flags=cv2.INTER_LINEAR)

camera = Camera()

fig = plt.figure(figsize=(12, 7))
a=fig.add_subplot(1,2,1)
img = cv2.imread('camera_cal/calibration1.jpg')
imgplot = plt.imshow(img)
a.set_title('Original Image')
a=fig.add_subplot(1,2,2)
dst = camera.undistort(img)
imgplot = plt.imshow(dst)
a.set_title('Undistorted Image')
plt.savefig('./output_images/undistorted_img.png')
plt.show()


# In[95]:


fig = plt.figure(figsize=(12, 7))
a=fig.add_subplot(1,2,1)
img = cv2.imread('./test_images/test4.jpg')[:, :, ::-1]
imgplot = plt.imshow(img)
a.set_title('original Image')
a=fig.add_subplot(1,2,2)
undist = camera.undistort(cv2.imread('./test_images/test4.jpg')[:, :, ::-1])
imgplot = plt.imshow(undist)
a.set_title('Undistorted Image')
plt.savefig('./output_images/test_undistorted_img.png')
plt.show()


# In[98]:


def filter_img(img):
    kernel0 = np.float32( [ [-1, -1, 0, 1, 1] ] )
    kernel1 = np.float32( [ [1, 1, 0, -1, -1] ] )
    smooth_img = cv2.GaussianBlur(img, (3, 3), 0)

    filterimg0 = cv2.filter2D(smooth_img, -1, kernel0)
    filterimg1 = cv2.filter2D(smooth_img, -1, kernel1)
    test0 = np.uint8(filterimg0)
    test1 = np.uint8(filterimg1)

    rst = cv2.bitwise_or(test0, test1)
    
    return rst

def color_filter(in_img):
    hls = cv2.cvtColor(in_img, cv2.COLOR_RGB2HLS)
    l_ch = hls[:,:,1]
    s_ch = hls[:,:,2]

    filter_0 = filter_img(l_ch)
    filter_1 = filter_img(s_ch)
    
    mask0 = cv2.bitwise_or(filter_0, filter_1)
    ret, mask00 = cv2.threshold(mask0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
       
    white_mask = cv2.inRange(
        hls,
        np.uint8([0, 200, 0]),
        np.uint8([255, 255, 255]),
    )
    yellow_mask = cv2.inRange(
        hls,
        np.uint8([10, 0, 100]),
        np.uint8([40, 255, 255]),
    )
    mask1 = cv2.bitwise_or(white_mask, yellow_mask)
    
    rtn_img = cv2.bitwise_or(mask00, mask1)
    
    return mask00, mask1, rtn_img

i = 0

for imgpath in glob('./test_images/*.jpg'):
    
    img = camera.undistort(imread_rgb(imgpath))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    f.set_figwidth(14)
    ax1.imshow(img)
    ax1.set_title('Original')
    simple_filtered_img, filtered_color_img, combine = color_filter(img)
    ax2.imshow(simple_filtered_img, cmap='gray')
    ax2.set_title('simple_filtered_img')
    ax3.imshow(filtered_color_img, cmap='gray')
    ax3.set_title('filtered_color_img')
    ax4.imshow(combine, cmap='gray')
    ax4.set_title('combine')
    out_file = './output_images/filter_result_img_%d.png' % i
    plt.savefig(out_file)
    i+=1
    plt.show()


# In[101]:


pts = np.array([(220, 720), 
                (570, 470), 
                (720, 470),
                (1110, 720)])
pts = pts.reshape((-1,1,2))
dst_pts = np.array([(370, 720), 
                  (370, 0), 
                  (870, 0),
                  (870, 720)])
dst_pts = dst_pts.reshape((-1,1,2))

fig = plt.figure(figsize=(12, 7))
a=fig.add_subplot(1,2,1)
img = camera.undistort(cv2.imread('./test_images/straight_lines2.jpg')[:, :, ::-1])
img_cp = np.copy(img)
cv2.polylines(img_cp, [pts], True, (255,0,0), 5)
imgplot = plt.imshow(img_cp)
a.set_title('undistorted Image')
a=fig.add_subplot(1,2,2)
dst = camera.warp(img)
dst_cp = np.copy(dst)
cv2.polylines(dst_cp, [dst_pts], True, (255,0,0), 5)
imgplot = plt.imshow(dst_cp)
a.set_title('warped Image')
plt.savefig('./output_images/perspectiveNInverse.png')
plt.show()


# In[102]:


i = 0
for imgpath in glob('./test_images/*.jpg'):
    img = camera.undistort(imread_rgb(imgpath))
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    f.set_figwidth(14)
    ax1.imshow(img)
    ax1.set_title('Original')
    simple_filtered_img, filtered_color_img, combine = color_filter(img)
    ax2.imshow(filtered_color_img, cmap='gray')
    ax2.set_title('Filtered')
    warped_img = camera.warp(filtered_color_img)
    ax3.imshow(warped_img, cmap='gray')
    ax3.set_title('Warped Binary')
    histogram = np.sum(warped_img[360:,:], axis=0)
    ax4.set_title('Historgram')
    ax4.plot(histogram)
    out_file = './output_images/warped_histogram_%d.png' % i
    plt.savefig(out_file)
    i+=1
    plt.show()


# In[103]:


def sliding_search(binary_warped, plot=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[360:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    else:
        left_fit = None
        left_fitx = None

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        right_fix = None
        right_fitx = None

    if plot:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img
    return left_fit, right_fit, left_fitx, right_fitx, ploty

i=0
for imgpath in glob('./test_images/*.jpg'):
    img = camera.undistort(imread_rgb(imgpath))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    f.set_figwidth(14)
    ax1.imshow(img)
    ax1.set_title('Original')
    simple_filtered_img, filtered_color_img, combine = color_filter(img)
    warped = camera.warp(filtered_color_img)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Warped Binary')
    left_fit, right_fit, lfx, rfx, ploty, lanes = sliding_search(warped, plot=True)
    ax3.imshow(lanes, cmap='gray')
    ax3.plot(lfx, ploty, color='yellow')
    ax3.plot(rfx, ploty, color='yellow')
    ax3.set_title('Sliding Windows')
    out_file = './output_images/sliding_window_%d' % i
    plt.savefig(out_file)
    i+=1
    plt.show()


# In[72]:


# Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
def margin_search(binary_warped, left_fit, right_fit, plot=True):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    else:
        left_fit = None
        left_fitx = None

    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        right_fix = None
        right_fitx = None
    
    if plot:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0] # Red
        out_img[righty, rightx] = [0, 0, 255] # Blue

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
    return left_fit, right_fit, left_fitx, right_fitx, ploty

img = camera.undistort(cv2.imread('./test_images/straight_lines1.jpg')[:, :, ::-1])
simple_filtered_img, filtered_color_img, combine = color_filter(img)
warped = camera.warp(filtered_color_img)
left_fit, right_fit, lfx, rfx, ploty = sliding_search(warped, plot=False)
left_fit, right_fit, lfx, rfx, ploty = margin_search(warped, left_fit, right_fit, plot=True)


# In[104]:


def calculate_curvature_centeroffset(left_fitx, right_fitx, ploty):
    lane_width = right_fitx[-1] - left_fitx[-1]
    x_mpp = 3.7/lane_width  # 3.7m lane width
    y_mpp = 30.0/720 # 30m for the whole height of the warped binary
    lane_center = (left_fitx[-1] + right_fitx[-1]) / 2 # Lane center in pixels
    center_offset = (1280 / 2 - lane_center) * x_mpp # offset from center converted to meters
    
    left_fitxm = np.polyfit(ploty * y_mpp, left_fitx * x_mpp, 2)
    right_fitxm = np.polyfit(ploty * y_mpp, right_fitx * x_mpp, 2)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fitxm[0]*y_eval*y_mpp + left_fitxm[1])**2)**1.5) / np.absolute(2*left_fitxm[0])
    right_curverad = ((1 + (2*right_fitxm[0]*y_eval*y_mpp + right_fitxm[1])**2)**1.5) / np.absolute(2*right_fitxm[0])
    return left_curverad, right_curverad, center_offset

def draw_lanes(undist, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp

def plot_back(undist, left_fitx, right_fitx, ploty):
    color_warp = draw_lanes(undist, left_fitx, right_fitx, ploty)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = camera.unwarp(color_warp)
    # Combine the result with the original image
    output = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Annotate curvature and deviation
    left_curve, right_curve, center_offset = calculate_curvature_centeroffset(left_fitx, right_fitx, ploty)
    cv2.putText(output, 'Left curvature: {:.3f}m'.format(left_curve), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, 'Right curvature: {:.3f}m'.format(right_curve), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, 'center_offset: {:.3f}m'.format(center_offset), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return output

i=0
for imgpath in glob('./test_images/*.jpg'):
    img = camera.undistort(imread_rgb(imgpath))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    f.set_figwidth(21)
    ax1.imshow(img)
    ax1.set_title('Original')
    simple_filtered_img, filtered_color_img, combine = color_filter(img)
    warped = camera.warp(filtered_color_img)
    left_x, right_x, lfx, rfx, ploty, lanes = sliding_search(warped, plot=True)
    ax2.imshow(lanes, cmap='gray')
    ax2.plot(lfx, ploty, color='yellow')
    ax2.plot(rfx, ploty, color='yellow')
    ax2.set_title('Sliding Windows')
    ax3.imshow(plot_back(img, lfx, rfx, ploty))
    ax3.set_title('Result')
    outfile = './output_images/result_%d.png' % i
    plt.savefig(outfile)
    i+=1
    plt.show()


# In[106]:


alpha = 0.25
max_diff = 50

avg_lfx, avg_rfx, ploty = None, None, ploty
skip_frame = False

count = 0

current_left_fit = []
current_right_fit = []

def process_frame_avg(img):
    global count
    global current_left_fit
    global current_right_fit
    global skip_frame
    img = camera.undistort(img)
    if not skip_frame:
        global avg_lfx
        global avg_rfx
        global ploty
        
        simple_filtered_img, filtered_color_img, combine = color_filter(img)
        warped = camera.warp(filtered_color_img)
        if (count == 0):
            left_fit, right_fit, lfx, rfx, ploty = sliding_search(warped)
            current_left_fit.append(left_fit)
            current_right_fit.append(right_fit)
        else:
            previous_left_fit = current_left_fit[-1]
            previous_right_fit = current_right_fit[-1]
            left_fit, right_fit, lfx, rfx, ploty = margin_search(warped, previous_left_fit, previous_right_fit, plot=False)
        
        if avg_lfx is None:
            avg_lfx = lfx
        if avg_rfx is None:
            avg_rfx = rfx

        if lfx is not None:
            diff = np.average(np.abs(avg_lfx - lfx))
            if diff < max_diff: # Drop and using previous diff to estimate position
                avg_lfx = alpha * lfx + (1 - alpha) * avg_lfx
        if rfx is not None:
            diff = np.average(np.abs(avg_rfx - rfx))
            if diff < max_diff:
                avg_rfx = alpha * rfx + (1 - alpha) * avg_rfx

    skip_frame = not skip_frame
    count = count + 1
    if(count == 11):
        count = 0
    return plot_back(img, avg_lfx, avg_rfx, ploty)

input_video = 'challenge_video.mp4'
output_video = 'output_{}'.format(input_video)

clip = VideoFileClip(input_video)
clip.fl_image(process_frame_avg).write_videofile(output_video, audio=False)


# In[ ]:




