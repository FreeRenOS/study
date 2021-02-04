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


#for picNum in range(22, 23): # 22번이 가장 에러가 많음(기준 이미지 설정)
for picNum in range(1, 31):

#for picNum in range(1, 6):
#for picNum in range(6, 11):
#for picNum in range(11, 16):
#for picNum in range(16, 21):
#for picNum in range(21, 26):
#for picNum in range(26, 31):

    # for DataFrame
    forDfList = []
    # 이미지 로드 및 프로세싱
    image = imread('./04_top/' + str(picNum) + '.bmp')
    image = FitToWindowSize(image) # 29.bmp 기준 shape = (1080, 1440, 3)
    imageCopy = image.copy() 
    imageGray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)

    ######################### 최외곽 네모 ###########################
    _, imageBinary = cv2.threshold(imageGray, 75, 255, type = cv2.THRESH_TOZERO_INV) # 이진화

    imageCanny = cv2.Laplacian(imageBinary, cv2.CV_8U)
    # imageCanny = cv2.Canny(imageBinary, 127, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imageMorph = cv2.morphologyEx(imageCanny, cv2.MORPH_CLOSE, kernel, iterations = 1)

    contours, _ = cv2.findContours(imageMorph, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # 테두리 찾기

    maxContours = max(contours, key = cv2.contourArea) # 가장 큰면적 

    rect = cv2.minAreaRect(maxContours) # minAreaRect
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(imageCopy, [box], 0, (0, 255, 0), 1) # 테두리 그리기
    for i in range(box.shape[0]):
        # 좌표 찍기, 텍스트 삽입 /// 참고: box.shape(4, 2) / box.dtype = np.uint64
        cv2.circle(imageCopy, tuple(box[i]), 5, (0, 255, 255), cv2.FILLED, cv2.LINE_4)
        cv2.putText(imageCopy, str(box[i][0]) + " : " + str(box[i][1]), tuple(box[i]), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
        forDfList.append(list(box[i])) # 데이터프레임 생성용 리스트

    # 좌표 DataFrame에 담기 / 0(우하), 1(좌하), 2(좌상), 3(우상), 4(우하)
    rectDf = pd.DataFrame(forDfList, columns = ['x', 'y'])
    tmp = list(rectDf.loc[0])
    rectDf = rectDf.T
    rectDf['4'] = tmp
    rectDf = rectDf.T
    # 좌표 간 연산 수행 및 중점 잡아서 텍스트 넣기
    distanceList = []
    for i in range(4):
        front_x = rectDf.iloc[i, 0]; behind_x = rectDf.iloc[i + 1, 0]
        front_y = rectDf.iloc[i, 1]; behind_y = rectDf.iloc[i + 1, 1]
        minX = min(front_x, behind_x)
        minY = min(front_y, behind_y)
        x = np.abs(front_x - behind_x)
        y = np.abs(front_y - behind_y)
        distance = np.sqrt(np.power(x, 2) + np.power(y, 2))
        distance = np.round(distance/100, 2)
        cv2.putText(imageCopy, str(distance), (int(minX + x/2), int(minY + y/2)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
    # distanceList = [8.33, 8.34, 8.33, 8.34]
        distanceList.append(distance)
    # 혹시 모를 재사용을 위해 원본으로 복원
    rectDf.drop(['4'], axis = 0)
    ################################################################
    
    ######################## 원 검출 ######################
    _, imageBinary2 = cv2.threshold(imageGray, 90, 255, type = cv2.THRESH_BINARY_INV) # 이진화
    # 가장 외곽에 있는 원
    # 가장 큰 원을 어떠한 빈 화면에 그리되 안쪽을 가득 채워서 그린 다음, 컨투어를 잡고 나서 그 컨투어의 무게중심을 imageCopy에 삽입하여 점을 찍어야 합니다.
    # imageCopy.shape = (1080, 1440, 3)
    circleZeros = np.ones(imageCopy.shape)

    bigCircles = cv2.HoughCircles(imageBinary2, cv2.HOUGH_GRADIENT, 1, 900, param1 = 150, param2 = 5, minRadius = 320, maxRadius = 325)
    for i in bigCircles[0]:
       cv2.circle(circleZeros, (i[0], i[1]), int(i[2]), (0, 255, 255), -1)

    #circleContours, _ = cv2.findContours(circleZeros, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #maxCircleContour = max(circleContours, key = cv2.contourArea)
    #cv2.drawContours(circleZeros, [maxCircleContour], 0, (0, 255, 255), 2)

    cv2.imshow('circleZeros', circleZeros)

    
    # 원의 모멘트 구하기
    
    



    for circle in bigCircles:
        M = cv2.moments(circle)
        cX = int(M['m10'] / M['m00'] + 1e-5)
        cY = int(M['m01'] / M['m00'] + 1e-5)

    print(cX, cY)
    cv2.circle(imageCopy, (cX, cY), 2, (255, 0, 255), -1)




    # 테두리 검출

    # _, imageBinary = cv2.threshold(imageGray, 90, 255, type = cv2.THRESH_BINARY_INV) # 이진화
    # contours, _ = cv2.findContours(imageBinary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # 테두리 찾기

    # for contour in contours:
    #     cv2.drawContours(imageCopy, [contour], 0, (255, 255, 0), 5)
    #     _area = cv2.contourArea(contour)
    #     forDfList.append([_area, contour])

    # mixedDf = pd.DataFrame(forDfList, columns = ['contour', 'area'])
    # display(mixedDf.head(3))
    # mixedDf.sort_values('contour', ascending = False, inplace = True)

    # cv2.drawContours(imageCopy, [mixedDf.iloc[1, 1]], 0, (0, 0, 255), 5)

    cv2.putText(imageCopy, str(picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('imageGray', imageGray)
    cv2.imshow('imageBinary', imageBinary)
    cv2.imshow('imageCanny', imageCanny)
    cv2.imshow('imageMorph', imageMorph)
    cv2.imshow('imageCopy', imageCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()