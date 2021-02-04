import numpy as np
import cv2
import pandas as pd
from win32api import GetSystemMetrics

for picNum in range(1, 12):
    img = cv2.imread("01_connecter_1/"+str(picNum)+".bmp", cv2.IMREAD_COLOR)

    # 직선과 점 사이의 거리 측정 함수
    def dist(P, A, B):
        area = abs ( (A[0] - P[0]) * (B[1] - P[1]) - (A[1] - P[1]) * (B[0] - P[0]) )
        AB = ( (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 ) ** 0.5
        return ( area / AB )

    img2 = img.copy()

    cv2.putText(img2, str(picNum), (50,50),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    # 그레이 스케일로 변환 및 엣지 검출 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150 )
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # 확율 허프 변환 적용
    lines = cv2.HoughLinesP(edges, 0.5, np.pi/180, 1, None, 1, 1) # 최대한 많이 잡음

    lineList = []
    for line in lines:
        lineList.append(line[0]) # 허프라인들의 좌표 저장

    listColumns = ['x1','y1','x2','y2']
    lineDF = pd.DataFrame(lineList, columns=listColumns)
    # x최소 y최소
    p1 = lineDF.sort_values(by=['x1'])
    p1 = p1[p1.x1 < p1.iloc[0,0]+15]
    p1 = p1.sort_values(by=['y1'])
    # x최소 y최대
    p2 = lineDF.sort_values(by=['x1'])
    p2 = p2[p2.x1 < p2.iloc[0,0]+15]
    p2 = p2.sort_values(by=['y1'], ascending=False)
    # x최대 y최소
    p4 = lineDF.sort_values(by=['x1'], ascending=False)
    p4 = p4[p4.x1 > p4.iloc[0,0]-15]
    p4 = p4.sort_values(by=['y1'], ascending=False)
    # x최대 y최대
    p3 = lineDF.sort_values(by=['x1'], ascending=False)
    p3 = p3[p3.x1 > p3.iloc[0,0]-15]
    p3 = p3.sort_values(by=['y1'])

    p1_x = p1.iloc[0,0]
    p1_y = p1.iloc[0,1]
    p2_x = p2.iloc[0,0]
    p2_y = p2.iloc[0,1]
    p3_x = p3.iloc[0,0]
    p3_y = p3.iloc[0,1]
    p4_x = p4.iloc[0,0]
    p4_y = p4.iloc[0,1]

    # dist함수에 쓰기위한 중심선의 시작점과 끝점
    cLineX = [p1_x, (p1_y + p2_y)/2]
    cLineY = [p3_x, (p3_y + p4_y)/2]

    # 사각형 그리기
    cv2.line(img2, (p1.iloc[0,0], p1.iloc[0,1]), (p2.iloc[0,0], p2.iloc[0,1]), (0,0,255), 2)
    cv2.line(img2, (p2.iloc[0,0], p2.iloc[0,1]), (p4.iloc[0,0], p4.iloc[0,1]), (0,0,255), 2)
    cv2.line(img2, (p4.iloc[0,0], p4.iloc[0,1]), (p3.iloc[0,0], p3.iloc[0,1]), (0,0,255), 2)
    cv2.line(img2, (p3.iloc[0,0], p3.iloc[0,1]), (p1.iloc[0,0], p1.iloc[0,1]), (0,0,255), 2)
    # 사각형의 가로 중심선, dist함수에서 비교할 선
    cv2.line(img2, (int(p1_x), int((p1_y + p2_y)/2)), (int(p3_x), int((p3_y + p4_y)/2)), (0,255,0),2)

    # 컨투어 그리기
    contourList=[] # 넘버링을 위한 리스트
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1350 : # 컨투어의 크기를 비교
            M = cv2.moments(cnt)
            cX = int(M['m10'] / M['m00'] + 1e-5)
            cY = int(M['m01'] / M['m00'] + 1e-5)
            p = [cX, cY] # dist함수에서 비교할 점
            d = dist(p, cLineX, cLineY) # 잡힌 무게중심와 사각형 중심선의 거리
            if d < 130 : # 거리를 비교
                contourList.append(p)
                cv2.drawContours(img2, [cnt], 0, (0,255, 0), 2)
                cv2.circle(img2, (cX, cY), 2, (255, 0, 0), -1) 
                # 경계사각형
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img2,[box],0,(255,127,127),2)

    contourDF = pd.DataFrame(contourList, columns=['x','y']) 
    contourDF = contourDF.sort_values(by=['x', 'y'])
    contourUP = contourDF[contourDF.y < (cLineX[1]+cLineY[1])/2] # 중심선 위에 있는 모멘트
    contourDOWN = contourDF[contourDF.y > (cLineX[1]+cLineY[1])/2] # 중심선 아래에 있는 모멘트

    # 넘버링
    for i in range(len(contourUP)):
        cv2.putText(img2, str(i+1), (contourUP.iloc[i,0],contourUP.iloc[i,1]-30),cv2.FONT_HERSHEY_COMPLEX, 1, (255,127,127), 2)
    for i in range(len(contourDOWN)):
        cv2.putText(img2, str(i+1), (contourDOWN.iloc[i,0],contourDOWN.iloc[i,1]-30),cv2.FONT_HERSHEY_COMPLEX, 1, (255,127,127), 2)

    #이미지 크기
    (h, w) = img2.shape[:2]

    #위도우 크기
    x = GetSystemMetrics(0) / w
    y = GetSystemMetrics(1) / h

    if x<y:
        img2 = cv2.resize(img2, (0,0), fx=x, fy=x)
    else:
        img2 = cv2.resize(img2, (0,0), fx=y, fy=y)

    cv2.imshow("img", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()