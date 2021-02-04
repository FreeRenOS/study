import cv2
from win32api import GetSystemMetrics
import numpy as np
import pandas as pd
print("Width =", GetSystemMetrics(0))
print("Height =", GetSystemMetrics(1))

# 파일명 한글 허용 함수
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

# 이미지 크기 조절 함수
def resizeImage(image):
    (h, w) = image.shape[:2]
    x = GetSystemMetrics(0) / w
    y = GetSystemMetrics(1) / h
    if x<y:
        image = cv2.resize(image, (0,0), fx=x, fy=x)
    else:
        image = cv2.resize(image, (0,0), fx=y, fy=y)
    return image

image = imread("03_connector_2-02/본딩 - 유무/1. 원본.bmp")

img_h = image.shape[0]
img_w = image.shape[1]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.rectangle(gray, (int(img_w/2+50), 0), (int(img_w), int(img_h)), (255,255,255),-1 )


ret, binary = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
binary2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations = 2)
contours, hierarchy = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('gray', gray)
cv2.imshow('gray2', binary2)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = max(contours, key=cv2.contourArea) # 최대크기 컨투어
print(c)
print(c.shape)
maxContourDF = c.reshape(len(c), 2)
print(maxContourDF)
maxContourDF = pd.DataFrame(maxContourDF, columns=['x','y']).sort_values(by='y') # 최대크기 컨투어 좌표 DF화 및 y값으로 정렬

pointsUp = maxContourDF[maxContourDF.y < maxContourDF.y.iloc[0]+30] # 상단 컨투어
pointsDown = maxContourDF[maxContourDF.y > maxContourDF.y.iloc[-1]-30] # 하단 컨투어

pointsUpLeft = pointsUp[pointsUp.x < pointsUp.x.mean()] # 좌상단
pointsUpRight = pointsUp[pointsUp.x > pointsUp.x.mean()] # 우상단
pointsDownLeft = pointsDown[pointsDown.x < pointsDown.x.mean()] # 좌하단
pointsDownRight = pointsDown[pointsDown.x > pointsDown.x.mean()] # 우하단

# 모서리
points = [ [pointsUpLeft.x.min(), int(pointsUpLeft.y.mean())],
        [pointsUpRight.x.max(), int(pointsUpRight.y.mean())], 
        [pointsDownRight.x.max(), int(pointsDownRight.y.mean())],
        [pointsDownLeft.x.min(), int(pointsDownLeft.y.mean())] ]

# 사각형 그리기
cv2.polylines(image,[np.array(points)], True, (0,255,255), 2)

# 사각형의 센터라인 양 끝 좌표
centerLine = [ [int((points[0][0] + points[1][0])/2), int((points[0][1] + points[1][1])/2)], 
            [int((points[2][0] + points[3][0])/2), int((points[2][1] + points[3][1])/2)] ]
# 센터라인 그리기
cv2.line(image, tuple(centerLine[0]), tuple(centerLine[1]), (0,255,255), 2)

# 선과 점 사이의 최소 거리 구하는 함수
def dist(P, A=centerLine[0], B=centerLine[1]):
    area = abs ( (A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]) )
    AB = ( (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 ) ** 0.5
    return ( area / AB )

# 모멘트 함수
def getMoments(contour):
    M = cv2.moments(cnt)
    x = int(M['m10'] / (M['m00'] + 1e-5))
    y = int(M['m01'] / (M['m00'] + 1e-5))
    return [x, y]

momentsList = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 300 and area < 2200: # 컨투어의 크기 비교
        p = getMoments(cnt)
        d = dist(p)
        cv2.drawContours(image, [cnt], 0, (0,255, 0), 2)
        cv2.circle(image, tuple(p), 2, (255, 0, 0), -1)
        if d <100: # 모멘트와 센터라인의 거리 비교
            momentsList.append(p)

momentsListDF = pd.DataFrame(momentsList, columns=['x','y'])
momentsLeft = momentsListDF[momentsListDF.x < momentsListDF.x.mean()].sort_values(by='y') # 좌측 모멘트
momentsRight = momentsListDF[momentsListDF.x > momentsListDF.x.mean()].sort_values(by='y') # 우측 모멘트

# 모멘트 사각형 모서리 좌표
momentsPoints = [ list(momentsLeft.iloc[0]),
                list(momentsLeft.iloc[-1]),
                list(momentsRight.iloc[-1]),
                list(momentsRight.iloc[0])]

# 모멘트 사각형 그리기
cv2.polylines(image,[np.array(momentsPoints)], True, (255,255,0), 2)

# 좌측 모멘트 넘버링
for i in range(len(momentsLeft)):
    cv2.putText(image, str(i+1), (momentsLeft.iloc[i,0]+20,momentsLeft.iloc[i,1]+10),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255, 0), 2)
# 우측 모멘트 넘버링
for i in range(len(momentsRight)):
    cv2.putText(image, str(i+1+len(momentsLeft)), (momentsRight.iloc[i,0]-60,momentsRight.iloc[i,1]+10),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255, 0), 2)

image = resizeImage(image)
cv2.imshow("img", image)
# cv2.imshow("gray2", gray)
# cv2.imshow("binary", binary2)
cv2.waitKey(0)
cv2.destroyAllWindows()