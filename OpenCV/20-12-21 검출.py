import sys
import numpy as np
import pandas as pd
import cv2
from win32api import GetSystemMetrics

## 한글 포함된 이미지 파일 읽어오기
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try :
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img

    except Exception as e:
        print(e)
        return None

def FitToWindowSize(image):
    image_resized = image.copy()
    #윈도우 크기 얻기
    win_w=GetSystemMetrics(0)
    win_h=GetSystemMetrics(1)
    img_h, img_w = image.shape[:2]

    if(img_h > win_h or img_w > win_w):   
        rate_width =  (win_w / img_w)*0.95
        rate_height =  (win_h / img_h)*0.95
        scale = rate_width if (rate_width < rate_height) else rate_height
        image_resized = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return image_resized

img = imread("03_connector_2-01/본딩 - 넘침/1. 원본.bmp")
img2 = img.copy()

img2 = FitToWindowSize(img2)
img3 = img2.copy()
img4 = img2.copy()

# 그레이 스케일로 변환 및 엣지 검출 
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,90, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(binary,100,200,apertureSize = 3)

## 코너 검출
corners = cv2.goodFeaturesToTrack(edges, 3000, 0.05, 1, blockSize=3, useHarrisDetector=True, k=0.03)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img3,(x,y),3,255,-1)

# ## 모폴로지 변환
# kernelsize = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# img_gray = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernelsize, iterations=1)
# ## 오픈(침식연산(erosion)-> 확장연산(dilation)) 검은색 늘어남 / 클로즈(확장연산-> 침식연산) 흰색이 늘어남

# _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
contours,_ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


for i in range(len(contours)):
    cv2.drawContours(img4, contours[i], -1, (0, 255, 0), 3)  

# 확율 허프 변환 적용
lines = cv2.HoughLinesP(edges, 0.8, np.pi/270, 90, 100, 10) # 최대한 많이 잡음
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),3)


# edges = FitToWindowSize(edges)

# cv2.imshow('img_gray', img_gray)

cv2.imshow('edges', edges)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)

cv2.waitKey(0)
cv2.destroyAllWindows()