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
    # 윈도우 크기 얻기
    # from win32api import GetSystemMetrics
    # print("Width =", GetSystemMetrics(0))
    # print("Height =", GetSystemMetrics(1))
    # 이미지 크기 얻기
    print('image {}'.format(image.shape))
    win_w = GetSystemMetrics(0)
    win_h = GetSystemMetrics(1)
    img_h, img_w = image.shape[:2]

    if(img_h > win_h or img_w > win_w):
        rate_width = (win_w / img_w)*0.95
        rate_height = (win_h / img_h)*0.95
        scale = rate_width if (rate_width < rate_height) else rate_height
        image_resized = cv2.resize(image, dsize=(
            0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('image_resize',image_resized)
    return image_resized

class D1218():
    def __init__(self):
        self.center =[]
        self.Big = []
        self.maxwh = 0
        self.RECT=[]
        self.MMT = []

    def drawconT(self, inputimg, image):
        #중심 점 찾기
        self.gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(self.gray,100, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(binary,100,200,apertureSize = 3)

        # cv2.imshow('edges', edges)

        # 코너 따기
        corners = cv2.goodFeaturesToTrack(edges, 3000, 0.05, 1, blockSize=3, useHarrisDetector=True, k=0.03)
        
        for i in corners:
            self.RECT.append(list(i[0]))
        self.RECT.sort(key = lambda REC : REC[0])
        # 코너 좌측 우측 나누기
        self.left = self.RECT[:10]
        self.right = self.RECT[-10:]
        # y 값으로 다시 sorting
        self.left.sort(key = lambda REC : REC[1])
        self.right.sort(key = lambda REC : REC[1])
        self.RECT.clear()
        # 4개의 점 추출
        self.RECT.append(self.left[0])
        self.RECT.append(self.left[-1])
        self.RECT.append(self.right[-1])
        self.RECT.append(self.right[0])
        # 사각형 그리기
        cv2.line(image, tuple(self.RECT[0]),tuple(self.RECT[1]),(255,0,0))
        cv2.line(image, tuple(self.RECT[1]),tuple(self.RECT[2]),(255,0,0))   
        cv2.line(image, tuple(self.RECT[2]),tuple(self.RECT[3]),(255,0,0))   
        cv2.line(image, tuple(self.RECT[0]),tuple(self.RECT[3]),(255,0,0)) 
        # 중심점
        self.MMT.append(int(np.mean(self.RECT,axis=0)[0]))
        self.MMT.append(int(np.mean(self.RECT,axis=0)[1]))
        cv2.circle(image,tuple(self.MMT),2,(255,255,0),0)
  
        # contour 
        self.gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(self.gray, 130, 255, cv2.THRESH_TOZERO)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 550 and cv2.contourArea(contours[i]) > 100:
                x, y, w, h = cv2.boundingRect(contours[i])
                M = cv2.moments(contours[i])
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                if w > h+5 and np.abs(cY-self.MMT[1])<75:
                    if cY< image.shape[0]/2:
                        cv2.fillConvexPoly(image,contours[i],(0,0,255))
                    else :
                        cv2.fillConvexPoly(image,contours[i],(255,0,0))
                    #rect그리기
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    #중심 그리기
                    cv2.circle(image,(cX,cY),1,(0,255,0),1)
                    self.center.append([cY,cX])
        
        self.center.sort(reverse=True)
        self.downside = self.center[:19]
        self.upside = self.center[19:]
        self.downside.sort(key= lambda side : side[1])
        self.upside.sort(key= lambda side : side[1])

        for i in range(len(self.downside)):
            cv2.putText(image,str(i),(self.downside[i][1]-10,self.downside[i][0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
            cv2.putText(image,str(i),(self.upside[i][1]-10,self.upside[i][0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))

        cv2.line(image, (self.upside[0][1]-20,self.upside[0][0]),(self.upside[-1][1]+20,self.upside[-1][0]),(255,255,0))
        cv2.line(image, (self.downside[0][1]-20,self.downside[0][0]),(self.downside[-1][1]+20,self.downside[-1][0]),(255,255,0))
        
        return image


if __name__ == '__main__':
    
    # 하나만 비교

    # D2 = D1218()
    # src = imread("C:/Users/w/Desktop/OPENCV/01_connecter_1/1.bmp")
    # src = FitToWindowSize(src)
    # dst = src.copy()
    # output = D2.drawconT(src, dst)
    # cv2.imshow('dst', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    # 여러개 한꺼번에 비교
    for i in range(10):
        src = imread("C:/Users/w/Desktop/OPENCV/01_connecter_1/"+str(i+1)+".bmp")
        D2 = D1218()
        src = FitToWindowSize(src)
        dst = src.copy()
        output = D2.drawconT(src, dst)
        cv2.imshow('dst', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        