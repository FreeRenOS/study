import cv2
from win32api import GetSystemMetrics
import numpy as np
import math

for picNum in range(1,31):
    
    image = cv2.imread("04_top/"+str(picNum)+".bmp", cv2.IMREAD_COLOR)
    if image is None:
        print("이미지를 읽을 수 없습니다.")
        exit(1)

    (h, w) = image.shape[:2]

    x = GetSystemMetrics(0) / w *0.8
    y = GetSystemMetrics(1) / h*0.8

    if x<y:
        image = cv2.resize(image, (0,0), fx=x, fy=x)
    else:
        image = cv2.resize(image, (0,0), fx=y, fy=y)

    cv2.putText(image, str(picNum), (100,100),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    _, binary = cv2.threshold(laplacian, 60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
    binary2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,iterations = 5)
    contours,_ = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    c = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),1)

    for i in range(4):
        if i < 3:
            a = box[i][0] - box[i+1][0]
            b = box[i][1] - box[i+1][1]
            x = (box[i][0] + box[i+1][0])/2
            y = (box[i][1] + box[i+1][1])/2
        else :
            a = box[i][0] - box[0][0]
            b = box[i][1] - box[0][1]
            x = (box[i][0] + box[0][0])/2
            y = (box[i][1] + box[0][1])/2
        d = math.sqrt((a**2)+(b**2))
        cv2.putText(image, str(round(d, 2)), (int(x)+5,int(y)-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    for i in range(4):
        x = box[i][0]
        y = box[i][1]
        cv2.circle(image, (x, y),3, (0, 255, 255), 2)
        cv2.putText(image, str(x) + ", " + str(y), (x-5,y-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    def drawCircles(circles):
        for i in circles[0]:
            x = i[0]
            y = i[1]
            r = i[2]
            cv2.circle(image, (x, y), int(r), (255, 127, 0), 2)
            if r <120:
                cv2.circle(image, (x, y), 3, (255, 255, 0), 2)
                cv2.putText(image, str(x) + ", " + str(y), (int(x)+5, int(y)-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    def findCircles(circles, minRadius, maxRadius):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 800, param1 = 250, param2 = 10, minRadius = minRadius, maxRadius = maxRadius) 
        return circles
    
    circleList = [[50,95], [188,203], [220,248], [250,260]]
    if picNum == 5 or picNum == 6 or picNum == 21:
        circleList[1][1] = 208
    
    circles=[]
    for i, j in circleList:
        circles = findCircles(circles, i, j)
        drawCircles(circles)

   



    cv2.imshow("img", image)
    # cv2.imshow("gray", gray)
    # cv2.imshow("binary", binary)
    # cv2.imshow("binary2", binary2)
    # cv2.imshow("laplacian", laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()