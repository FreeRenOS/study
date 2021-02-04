import cv2
import numpy as np
from win32api import GetSystemMetrics 

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
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



filename="./03_connector_2-01/FPCB - 금속 그루터기 (Burr)/1. 원본.bmp"
origin=imread(filename)
dst=origin.copy()

img_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(img_gray,(3,3),0)
ret, thr = cv2.threshold(blurred,120,130,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
# ret1, thr1 = cv2.threshold(img_gray,110,130,cv2.THRESH_BINARY_INV)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# thr1 = cv2.morphologyEx(thr1, cv2.MORPH_CLOSE, kernel1, iterations=5)
edges=cv2.Canny(thr,50,100)
contours, _ = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c=max(contours,key=cv2.contourArea)
rect=cv2.minAreaRect(c)
box=cv2.boxPoints(rect)
box=np.int0(box)
# cv2.drawContours(dst,[box],0,(0,255,0),3)

lines=cv2.HoughLines(edges,1,np.pi/180,200,None,0,0)
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
    # cv2.line(dst,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)


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
interPoint1=[]
for a1,b1 in point:
    for a2,b2 in point:
        if a1!=a2:
            interX=(b2-b1)/(a1-a2)
            interY=a1*interX+b1
            interX=round(interX)
            interY=round(interY)

            if 0<interX<1000 and 0<interY<1944:
                interPoint.append([interX,interY])
                # cv2.circle(dst,(interX,interY),10,(0,255,0),4) 
            if 0<interX<700 and 0<interY<1944:
                interPoint1.append([interX,interY])
                # cv2.circle(dst,(interX,interY),10,(255,0,0),4)

print("1111111111111111111111111111111111111111111111111111111111111111")
print(interPoint)
print("333333333333333333333333333333333333333333333333333333333")
for i in interPoint:
    print(i)
    print(i[0])


def pointMaking(list,interPoint):
    interPointX=[]
    interPointY=[]
    interPointSum=[]
    for i,j in interPoint:
        interPointX.append(i)
        interPointY.append(j)
        sum=i+j
        interPointSum.append(sum)

    recPoint1=[]
    recPoint2=[]
    recPoint3=[]
    recPoint4=[]

    for i in interPoint:
        if min(interPointX)<=i[0]<=min(interPointX)+15 and max(interPointY)-15<=i[1]<=max(interPointY):
                recPoint1.append([i[0],i[1]])
        sum=i[0]+i[1]
        if sum == min(interPointSum):
           
            recPoint2.append([i[0],i[1]])

        if sum == max(interPointSum):
            recPoint3.append([i[0],i[1]])
        if max(interPointX)-15<=i[0]<=max(interPointX) and min(interPointY)<=i[1]<=min(interPointY)+25:
            recPoint4.append([i[0],i[1]])
    
    # print(recPoint1)
    # print(recPoint2)
    # print(recPoint3)
    # print(recPoint4)
    list.append(recPoint1)
    list.append(recPoint2)
    list.append(recPoint3)
    list.append(recPoint4)

print("22222222222222222222222222222222222222222222222222222222222")

listRecPoint0=[]
listRecPoint1=[]
pointMaking(listRecPoint0,interPoint)
pointMaking(listRecPoint1,interPoint1)
print(listRecPoint0)
print(listRecPoint1)
mX1=round(listRecPoint1[2][0][0]+((listRecPoint0[2][0][0]-listRecPoint1[2][0][0]))/2)
mX0=round(listRecPoint1[3][0][0]+((listRecPoint0[3][0][0]-listRecPoint1[3][0][0]))/2)

newX0=mX0+(mX0-listRecPoint0[1][0][0])+1
newX1=mX1+(mX1-listRecPoint1[0][0][0])+1
print("33333333333333333333333333333333333333333333333333333333333")

# y=ax+b

a0= (listRecPoint0[3][0][1]-listRecPoint0[1][0][1])/(listRecPoint0[3][0][0]-listRecPoint0[1][0][0])
b0= listRecPoint0[1][0][1]-(a0*listRecPoint0[1][0][0])
a1= (listRecPoint0[2][0][1]-listRecPoint0[0][0][1])/(listRecPoint0[2][0][0]-listRecPoint0[0][0][0])
b1= listRecPoint0[0][0][1]-(a1*listRecPoint0[0][0][0])

newY0=round(a0*newX0+b0)
newY1=round(a1*newX1+b1)
print(listRecPoint0[0][0])
print(listRecPoint0[1][0])
print(newX0,newY0)
print(newX1,newY1)
print("444444444444444444444444444444444444444444444444444444444")


newPoint0=[newX0,newY0]
newPoint1=[newX1,newY1]
cv2.circle(dst,(newX0,newY0),10,(0,255,0),2)
cv2.circle(dst,(newX1,newY1),10,(0,255,0),2)
cv2.circle(dst,(listRecPoint0[0][0][0],listRecPoint0[0][0][1]),10,(0,255,0),2)
cv2.circle(dst,(listRecPoint0[1][0][0],listRecPoint0[1][0][1]),10,(0,255,0),2)

cv2.line(dst,(newX0,newY0),(newX1,newY1),(0,0,255),2)
cv2.line(dst,(newX0,newY0),(listRecPoint0[1][0][0],listRecPoint0[1][0][1]),(0,0,255),2)
cv2.line(dst,(newX1,newY1),(listRecPoint0[0][0][0],listRecPoint0[0][0][1]),(0,0,255),2)
cv2.line(dst,(listRecPoint0[1][0][0],listRecPoint0[1][0][1]),(listRecPoint0[0][0][0],listRecPoint0[0][0][1]),(0,0,255),2)






contours1,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# print("33333333333333333333333333333333333333333333")
# print(np.array(contours1))
# print("4444444444444444444444444444444444444444444444444")
for i in contours1:
    if 300 < i[0][0][0] < 500:
        # cv2.drawContours(dst,i,-1,[0,255,0],3)
        
        print(np.array(i))


# contourPoints=[]
# for i in range(0,len(contours)):
#     for j in range(0,len(contours[i])):
#         contourPoints.append(contours[i][j])
# contourPoints=np.array(contourPoints)
# print("22222222222222222222222222222222222222222222222222222222222")
# print(contourPoints)
# ''


# for i in contours1:
#         cv2.drawContours(dst,i,-1,[0,0,255],3)
# cv2.circle(dst,(newX1,1000),10,(0,0,255),4)
# cv2.circle(dst,(615,19),10,(0,0,255),4)

edges=FitToWindowSize(edges)
dst=FitToWindowSize(dst)
thr=FitToWindowSize(thr)
# thr1=FitToWindowSize(thr1)
# cv2.imshow("edges",edges)
cv2.imshow("dst",dst)
# cv2.imshow("thr",thr)
# cv2.imshow("thr1",thr1)
cv2.waitKey(0)
cv2.destroyAllWindows(0)