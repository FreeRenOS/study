import cv2
import sys
from PyQt5 import QtWidgets, uic, QtGui
import cv2
import numpy as np
from win32api import GetSystemMetrics
from collections import defaultdict


def FitToWindowSize(image):
    image_resized = image.copy()
    #윈도우 크기 얻기
    # print('image {}'.format(image.shape))
    win_w=GetSystemMetrics(0)
    win_h=GetSystemMetrics(1)
    img_h, img_w = image.shape[:2]

    if(img_h > win_h or img_w > win_w):   
        rate_width =  (win_w / img_w)*0.85
        rate_height =  (win_h / img_h)*0.85
        scale = rate_width if (rate_width < rate_height) else rate_height
        image_resized = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('image_resize',image_resized)
    return image_resized    



# imageFile = "./images/PCB2/pcbImage/gate1_1.bmp"

for q in range(1,9):

    imageFile = "./images/01_connecter_1/{}.bmp".format(q)


    origin=cv2.imread(imageFile)
    dst=origin.copy()
    x=500
    y=500
    w=200
    h=100
    roi=dst[y:y+h,x:x+w]


    img_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    print("1111111111111111111111111111111")
    print(img_gray.shape)
    ret, thr = cv2.threshold(img_gray,160,180,cv2.THRESH_TOZERO)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=4)
    ret1, thr1 = cv2.threshold(img_gray,100,150,cv2.THRESH_TOZERO_INV)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thr1 = cv2.morphologyEx(thr1, cv2.MORPH_CLOSE, kernel1, iterations=4)

    contours, _ = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    boundingLoc=[]

    putNumber0=[]
    putColumn0=[]
    putNumber1=[]
    putColumn1=[]
    putNumberSorted0=[]
    putNumberSorted1=[]
    contours2, _ = cv2.findContours(thr1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours2:
        cv2.drawContours(dst,i,-1,[0,0,255],3)

    edges=cv2.Canny(thr1,100,200)
    lines=cv2.HoughLines(edges,1,np.pi/180,150,None,0,0)
    linePoints=[]
    for i in range(len(lines)):
        rho=lines[i][0][0]
        theta=lines[i][0][1]    
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+2000*(-b))
        y1=int(y0+2000*(a))
        x2=int(x0-2000*(-b))
        y2=int(y0-2000*(a))
        linePoints.append([x1,y1,x2,y2])
        # cv2.line(origin,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
    point=[]
    for line in linePoints:
        x1=line[0]
        y1=line[1]
        x2=line[2]
        y2=line[3]
        u=y2-y1
        z=x2-x1
        if z!=0:
            a=u/z
            b=y1-a*x1
            point.append([a,b])
        if z==0:
            a=u/(z+1e-5)
            b=y1-a*x1
            point.append([a,b])

    interPoint=[]
    for a1,b1 in point:
        for a2,b2 in point:
            if a1!=a2:
                interX=(b2-b1)/(a1-a2)
                interY=a1*interX+b1
                interX=round(interX)
                interY=round(interY)

                if 0<interX<2592 and 0<interY<1944:
                    interPoint.append([interX,interY])
                    # cv2.circle(origin,(interX,interY),10,(0,255,0),4) 

    print("7777777777777777777777777777777777777777")
    print(interPoint)
    xList=[]
    yList=[]

    for i,j in interPoint:
        xList.append(i)
        yList.append(j)
    print("88888888888888888888888888888888888888888")
  
    xMax=max(xList)
    xMin=min(xList)
    yMax=max(yList)
    yMin=min(yList)
    print(xMax)
    print(yMax)
    # cv2.circle(origin,(xMax,yMax),10,(0,255,0),4) 
    # cv2.circle(origin,(xMin,yMax),10,(0,255,0),4) 
    # cv2.circle(origin,(xMax,yMin),10,(0,255,0),4) 
    # cv2.circle(origin,(xMin,yMin),10,(0,255,0),4) 
    line1PointXr=xMax+30
    line1PointYr=yMax
    line1PointXl=xMin-30
    line1PointYl=yMax
    line2PointXup=xMin
    line2PointYup=yMin
    line2PointXdown=xMin
    line2PointYdown=yMax+30
    line3PointXup=xMax
    line3PointYup=yMin
    line3PointXdown=xMax
    line3PointYdown=yMax+30
    cv2.line(origin,(line1PointXl,line1PointYl),(line1PointXr,line1PointYr),(0,0,255),2)
    cv2.line(origin,(line2PointXup,line2PointYup),(line2PointXdown,line2PointYdown),(0,0,255),2)
    cv2.line(origin,(line3PointXup,line3PointYup),(line3PointXdown,line3PointYdown),(0,0,255),2)



    for i in range(len(contours)):

        area=cv2.contourArea(contours[i])
        # print(area)
        if 1330<area<1800:
            x,y,w,h=cv2.boundingRect(contours[i])
            boundingLoc.append([x,y,w,h])
            if h<48 and 805<y<1030:
                cv2.rectangle(origin,(x,y),(x+w,y+h),(0,255,0),2)
                # print("hhhhhhhhhhhhhhhhhh")
                # print(h)
                cv2.drawContours(origin,[contours[i]],-1,(0,255,255),-1)
                mu=cv2.moments(contours[i])
                cX=int(mu["m10"] / (mu["m00"]+1e-5))
                cY=int(mu["m01"] / (mu["m00"]+1e-5))
                putNumberSorted0.append([cX,cY])
                cv2.circle(origin,(cX,cY),4,(255,0,0),2)
                column = contours[i][0][0][0] #가로 좌표(x)0
                row = contours[i][0][0][1]    #세로 좌표(y)
                # cv2.putText(origin,str(area),(column-20,row-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
                # cv2.putText(img_rgb,str(i),tuple(contours[i][0][0]),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

                putNumber0.append([column,row])
                putColumn0.append(column)

                # cv2.putText(origin,str(count),(column,row-20),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2)

            if h<48 and 1030<y<1120:
                cv2.rectangle(origin,(x,y),(x+w,y+h),(0,255,0),2)
        
                cv2.drawContours(origin,[contours[i]],-1,(255,255,0),-1)
                mu=cv2.moments(contours[i])
                cX=int(mu["m10"] / (mu["m00"]+1e-5))
                cY=int(mu["m01"] / (mu["m00"]+1e-5))
                putNumberSorted1.append([cX,cY])
                cv2.circle(origin,(cX,cY),4,(255,0,0),2)
                column = contours[i][0][0][0] #가로 좌표(x)0
                row = contours[i][0][0][1]    #세로 좌표(y)
                # cv2.putText(origin,str(area),(column-20,row-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
                # cv2.putText(img_rgb,str(i),tuple(contours[i][0][0]),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
                putNumber1.append([column,row])
                putColumn1.append(column)
                # cv2.putText(origin,str(count1),(column,row+60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2)

    # cv2.circle(origin,(1262,1050),10,(0,0,255),4)
    print("222222222222222222222222222222222222")
    # print(boundingLoc)
    print("333333333333333333333333333333333333")
    # print(contours[0])

    putColumn0.sort()
    putColumn1.sort()
    print("44444444444444444444444444444444444444")
    print(putColumn0)

    count0=0
    for i in putColumn0:
        for j in putNumber0:
            if i==j[0]:
                count0=count0+1
                # putNumberSorted0.append([j[0],j[1]])
                cv2.putText(origin,str(count0),(j[0],j[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
    print("5555555555555555555555555555555555")
    print(putNumberSorted0)
    for i,j in zip(range(0,len(putNumberSorted0)-1),range(1,len(putNumberSorted0))):
        cv2.line(origin,(putNumberSorted0[i][0],putNumberSorted0[i][1]),(putNumberSorted0[j][0],putNumberSorted0[j][1]),(0,0,255),2)


    count1=0
    for i in putColumn1:
        for j in putNumber1:
            if i==j[0]:
                count1=count1+1
                # putNumberSorted1.append([j[0],j[1]])
                cv2.putText(origin,str(count1),(j[0],j[1]+60),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
    print("5555555555555555555555555555555555")
    print(putNumberSorted1)
    for i,j in zip(range(0,len(putNumberSorted1)-1),range(1,len(putNumberSorted1))):
        cv2.line(origin,(putNumberSorted1[i][0],putNumberSorted1[i][1]),(putNumberSorted1[j][0],putNumberSorted1[j][1]),(0,0,255),2)

    endValuePoint0=len(putNumberSorted0)-1
    endValuePoint1=len(putNumberSorted1)-1
    centerLinePoint0=(putNumberSorted0[0][0],putNumberSorted0[0][1]+round(((putNumberSorted1[0][1]-putNumberSorted0[0][1])/2)))
    centerLinePoint1=(putNumberSorted0[endValuePoint0][0],putNumberSorted0[endValuePoint0][1]+round(((putNumberSorted1[endValuePoint1][1]-putNumberSorted0[endValuePoint0][1])/2)))

    cv2.line(origin,centerLinePoint0,centerLinePoint1,(255,0,0),2)

    print("6666666666666666666666666666666666")
    print(putNumber0) 
    print(putNumber1)

    img_gray=FitToWindowSize(img_gray)
    thr=FitToWindowSize(thr)
    origin=FitToWindowSize(origin)
    dst=FitToWindowSize(dst)
    edges=FitToWindowSize(edges)
    # cv2.imshow("img_gray",img_gray)
    # cv2.imshow("thr",thr)
    # cv2.imshow("dst",dst)
    # cv2.imshow("edges",edges)
    cv2.imshow("origin",origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # contours,_ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(len(contours)):
    #     area=cv2.contourArea(contours[i])
        
    #     # if area > 200
    #     cv2.drawContours(img_rgb,[contours[i]],-1,(0,255,0),3)
    #     mu=cv2.moments(contours[i])
    #     cX=int(mu["m10"] / (mu["m00"]+1e-5))
    #     cY=int(mu["m01"] / (mu["m00"]+1e-5))
        
    #     cv2.circle(img_rgb,(cX,cY),10,(255,0,0),-1)
    #     column = contours[i][0][0][0] #가로 좌표(x)0
    #     row = contours[i][0][0][1]    #세로 좌표(y)
    #     cv2.putText(img_rgb,str(area),(column-20,row-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
    #     # cv2.putText(img_rgb,str(i),tuple(contours[i][0][0]),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
    #     cv2.putText(img_rgb,str(i)+"/",(column-60,row-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2)


    # origin=cv2.imread(imageFile)
    # dst=origin.copy()
    # img=cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
    # img_blur=cv2.medianBlur(img,5)
    # # cv2.namedWindow('contour',cv2.WINDOW_NORMAL)
    # # ret, thr = cv2.threshold(img,94,99,cv2.THRESH_TOZERO_INV)
    # ret, thr = cv2.threshold(img,94,99,cv2.THRESH_TOZERO_INV)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=4)
    # contours, _ = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     area=cv2.contourArea(contours[i])
        
    #     # if area > 200
    #     cv2.drawContours(origin,[contours[i]],-1,(0,255,0),3)

    # xContour=[]
    # yContour=[]
    # for i in contours:
    #     if i[0][0][0] > 200:
    #         xContour.append(i[0][0][0])
    #         yContour.append(i[0][0][1])

    # print("1111111111111111111111111111111")
    # print(xContour)
    # print(yContour)

    # contour=FitToWindowSize(dst)
    # cv2.imshow("contour",contour)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()