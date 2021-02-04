from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QSlider
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

# OPENCV 화면 출력 관련 함수 : 화면크기에 맞춰 이미지 출력
def FitToWindowSize(image):
    image_resized = image.copy()
    #윈도우 크기 얻기
    # print('image {}'.format(image.shape))
    win_w=GetSystemMetrics(0)
    win_h=GetSystemMetrics(1)
    img_h, img_w = image.shape[:2]

    if(img_h > win_h or img_w > win_w):   
        rate_width =  (win_w / img_w)*0.95
        rate_height =  (win_h / img_h)*0.95
        scale = rate_width if (rate_width < rate_height) else rate_height
        image_resized = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('image_resize',image_resized)
    return image_resized

for picNum in range(1, 12): # ~11

# 이미지 로드 및 프로세싱
    image = imread('./01_connecter_1/' + str(picNum) + '.bmp', flags = cv2.IMREAD_COLOR, dtype = np.uint8)
    image = FitToWindowSize(image)
    imageCopy = image.copy() # (1080, 1440, 3), dtype = uint8
    cv2.putText(imageCopy, str(picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)
    height, width, channel = imageCopy.shape # 1080 1440 3
    imageGray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)
    # 115, 255, cv2.THRESH_BINARY_INV: 모든 이미지에 가장 큰 네모가 잡히는 최저 임계지점
    _, imageBinary = cv2.threshold(imageGray, 115, 255, cv2.THRESH_BINARY_INV)
    # 60, 255, cv2.THRESH_BINARY_INV: 모든 사각형을 잡기 위해 적용되는 임계지점
    _, imageBinary2 = cv2.threshold(imageGray, 60, 255, cv2.THRESH_BINARY_INV)

# 변수 선언
    contourList = [] # 조건을 만족하는 컨투어를 담을 리스트
    momentList = [] # 모멘트 좌표(cX, cY)를 담을 리스트

## 큰 사각형
    contours, _ = cv2.findContours(imageBinary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    maxContour = max(contours, key = cv2.contourArea) # maxContour는 3차원 텐서이므로
    
    dfContour = np.array(maxContour).sum(axis = 1) # sum 함수를 통해 2차행렬로 변환합니다.
    print(dfContour)
    df = pd.DataFrame(dfContour, columns = ['x', 'y'])
    df = df.sort_values(by = ['x', 'y'])
    print(df)
    dfCopy = df.copy() # x, y로 솔팅된 데이터프레임 원본 유지
    dfXMin = dfCopy['x'].min();                   dfXMax = dfCopy['x'].max()
    # boolean indexing
    dfCopy = dfCopy[(dfCopy['x'] < dfXMin + 30) | (dfCopy['x'] > dfXMax - 30)]
    # X, Y 최대 최소값 구하기
    dfYMin = dfCopy['y'].min();                   dfYMax = dfCopy['y'].max()
    cv2.rectangle(imageCopy, (dfXMin, dfYMin), (dfXMax, dfYMax), (0, 255, 0), 1, cv2.LINE_4)
    # 네모에 대한 중간좌표 구하기
    centRectX = np.abs(dfXMax - dfXMin) / 2;      centRectY = np.abs(dfYMax - dfYMin) / 2
    centerX = dfXMin + int(centRectX);            centerY = dfYMin + int(centRectY)
    # 드로잉
    cv2.circle(imageCopy, (centerX, centerY), 1, (255, 0, 255), -1)

### 모멘트
    contours, _ = cv2.findContours(imageBinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if contours is None:
        print("error")
        break
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500 and area < 630:
            # 모멘트 구하기
            M = cv2.moments(contour)
            cX = int(M['m10'] / M['m00'] + 1e-5)
            cY = int(M['m01'] / M['m00'] + 1e-5)
            if cY > centerY - 75 and cY < centerY + 75:
                momentList.append([cX, cY])
                # 조건을 만족하는 컨투어 및 영역만 담겨져있는 리스트
                contourList.append(contour)
                if cY > centerY:
                    cv2.circle(imageCopy, (cX, cY), 2, (100, 0, 45), -1)
                else:
                    cv2.circle(imageCopy, (cX, cY), 2, (100, 0, 200), -1)            

#### 정렬
    momentDf = pd.DataFrame(momentList, index = np.arange(1, len(momentList) + 1), columns = ['momentX', 'momentY'])
    momentDf.sort_values(by = ['momentY', 'momentX'], inplace = True)
    momentDf.index = np.arange(1, len(momentList) + 1)
    upMomentDf = momentDf[:19] # 위쪽 네모 19개의 데이터프레임
    downMomentDf = momentDf[19:] # 아래쪽 네모 19개의 데이터프레임
    upMomentDf.sort_values(by = 'momentX', inplace = True)
    downMomentDf.sort_values(by = 'momentX', inplace = True)

##### 위 네모 및 아래 네모 19개를 각각 담은 데이터프레임을 가지고 순서를 부여합니다.
    for upItem in range(len(upMomentDf)):
        cv2.putText(imageCopy, str(upItem + 1), (upMomentDf.iloc[upItem, 0] - 10, upMomentDf.iloc[upItem, 1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (40, 150, 255), 1)
    for downItem in range(len(downMomentDf)):
        cv2.putText(imageCopy, str(downItem + 1), (downMomentDf.iloc[downItem, 0] - 10, downMomentDf.iloc[downItem, 1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (40, 150, 255), 1)
    for contour in contourList:
        # 사각형 그리기 위한 좌표 구하기
        minX, minY = np.min(contour, axis = (0, 1))
        maxX, maxY = np.max(contour, axis = (0, 1))
        cv2.rectangle(imageCopy, (minX, minY), (maxX, maxY), (50, 255, 255), 1, cv2.LINE_8)

    cv2.imshow('imageCopy', imageCopy)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
    else:
        cv2.destroyAllWindows()