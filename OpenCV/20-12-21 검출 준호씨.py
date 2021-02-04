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

class D1221():
    def __init__(self):
        self.center =[]
        self.Big = []
        self.maxwh = 0
        self.RECT=[]
        self.MMT = []
        self.RECT2 = []
        self.ANY = []
    # 큰 사각형 따기    
    def drawconT(self, inputimg, image):

        self.gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(self.gray,90, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(binary,100,200,apertureSize = 3)

        # 코너 따기
        corners = cv2.goodFeaturesToTrack(edges, 3000, 0.05, 1, blockSize=3, useHarrisDetector=True, k=0.03)
        for i in corners:
            self.RECT.append(list(i[0]))

        self.RECT.sort(key = lambda REC : REC[0])

        # 먼저 왼쪽 두 좌표 따기
        self.left = self.RECT[:20].copy()
        self.left.sort(key = lambda REC : REC[1])
        self.RECT.clear()
        self.RECT.append(self.left[0])
        self.RECT.append(self.left[-1])

        # 오른 쪽 두 좌표 따기
        for i in corners:
            if np.abs(i[0][0]-self.RECT[0][0]) <image.shape[1]/2:
                
                self.RECT2.append(list(i[0]))

        
        self.RECT2.sort(key = lambda REC :REC[1])
        upRight = self.RECT2[:30]
        downRight = self.RECT2[-30:]
        upRight.sort(key = lambda up : up[0])
        downRight.sort(key = lambda down : down[0])
        self.RECT.append(upRight[-1])
        self.RECT.append(downRight[-1])
        print(self.RECT)
        self.RECT2.clear()

       
        for i in corners:
            if i[0][0]-self.RECT[2][0] >0 and  i[0][0]-self.RECT[2][0]<100 and abs(i[0][1] - self.RECT[2][1])>50 and abs(i[0][1] - self.RECT[3][1])>50 :
                self.ANY.append(list(i[0]))
        
        self.ANY.sort(key = lambda REC : REC[1])
        self.RECT2.append(self.ANY[0])
        self.RECT2.append(self.ANY[-1])
        
        cv2.circle(image,tuple(self.RECT2[0]),4,(255,0,0))
        cv2.circle(image,tuple(self.RECT2[1]),4,(255,0,0))
        self.ANY.clear()

        
        image2 = image[:,int(self.RECT[2][0]):].copy()
        
        gray2 =  cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        _, binary2 = cv2.threshold(gray2,36, 255, cv2.THRESH_BINARY_INV)
        corners = cv2.goodFeaturesToTrack(binary2, 3000, 0.05, 1, blockSize=3, useHarrisDetector=True, k=0.03)
        for i in corners:
            self.ANY.append(list(i[0]))

        self.ANY.sort(key= lambda ANY : ANY[0])
        print(self.ANY)
        ANY2 = self.ANY[int(len(self.ANY)/2):]
        ANY2.sort(key = lambda ANY2 : ANY2[1])
        print(ANY2)
        X  = ANY2[0][0]
        image2 = image2[:,:int(X)].copy()
        gray2 =  cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        count = 0
        ret, binary2 = cv2.threshold(gray2,100, 255, cv2.THRESH_BINARY_INV)

        #오른쪽 contour
        kernel = np.ones((3,3),np.uint8)
        binary3 = cv2.erode(binary2,kernel,iterations=1)

        contours, __ = cv2.findContours(
            binary3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cv2.drawContours(image2,contours,i,(255,0,255),1)
        

        
        ret, binary2 = cv2.threshold(gray2,60, 255, cv2.THRESH_BINARY_INV)
        contours, __ = cv2.findContours(
            binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            M = cv2.moments(contours[i])
            if w*h < 50000 and w*h> 1000:
                count+=1

        if count == 17 :
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if w*h < 50000 and w*h> 1000:
                    cv2.rectangle(image2,(x,y),(x+w,y+h),(255,0,0),2)

        image[:,int(self.RECT[2][0]):int(self.RECT[2][0])+int(X)] = image2
        # 사각형 그리기
        cv2.line(image, tuple(self.RECT[0]),tuple(self.RECT[1]),(255,0,0))
        cv2.line(image, tuple(self.RECT[1]),tuple(self.RECT[3]),(255,0,0))   
        cv2.line(image, tuple(self.RECT[2]),tuple(self.RECT[3]),(255,0,0))   
        cv2.line(image, tuple(self.RECT[0]),tuple(self.RECT[2]),(255,0,0)) 
        # 중심점
        self.MMT.append(int(np.mean(self.RECT,axis=0)[0]))
        self.MMT.append(int(np.mean(self.RECT,axis=0)[1]))
        cv2.circle(image,tuple(self.MMT),4,(255,255,0),0)
        
        image = self.CenterCon(inputimg,image)
        if count != 17:
                print(count)
                cv2.putText(image, "Right Error", (800,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))

        return image

    #작은 사각형 따기
    def CenterCon(self,inputimg,image):
        self.gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
        self.sideCon = []
        _, binary = cv2.threshold(self.gray, 125, 255, cv2.THRESH_TOZERO)
        contours, __ = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 550 and cv2.contourArea(contours[i]) > 100:
                x, y, w, h = cv2.boundingRect(contours[i])
                M = cv2.moments(contours[i])
                cX = int(M['m10'] / M['m00']+ 1e-5)
                cY = int(M['m01'] / M['m00']+ 1e-5)
                if np.abs(cX-self.MMT[0])<60:
                    cv2.drawContours(image, contours,i,(255,255,0))
                    self.center.append([cX,cY])
                elif np.abs(cX-self.MMT[0])>60 and np.abs(cX-self.MMT[0])<90:
                    cv2.drawContours(image, contours,i,(0,255,255))
                    self.sideCon.append([cX,cY])
        
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        print(self.center)

        if len(self.center) != 38 :
            cv2.putText(image, "center RECT error", (100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            print(len(self.center))
        else:
            self.center.sort(reverse=True)
            print(self.center)
            self.rightside = self.center[:19]
            self.leftside = self.center[19:]
            for i in self.rightside:
                if i[0]<self.MMT[0]:
                    cv2.putText(image, "center RECT error", (100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            for i in self.leftside:
                if i[0]>self.MMT[0]:
                    cv2.putText(image, "center RECT error", (100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            self.rightside.sort(key= lambda side : side[1])
            self.leftside.sort(key= lambda side : side[1])
            for i in range(len(self.leftside)):
                cv2.putText(image,str(i+18),(self.rightside[i][0]-30,self.rightside[i][1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
                cv2.putText(image,str(i),(self.leftside[i][0]+10,self.leftside[i][1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))

            cv2.line(image, (self.leftside[0][0],self.leftside[0][1]),(self.leftside[-1][0],self.leftside[-1][1]),(255,255,0))
            cv2.line(image, (self.rightside[0][0],self.rightside[0][1]),(self.rightside[-1][0],self.rightside[-1][1]),(255,255,0))
            cv2.line(image, (self.leftside[0][0],self.leftside[0][1]),(self.rightside[0][0],self.rightside[0][1]),(255,255,0))
            cv2.line(image, (self.leftside[-1][0],self.leftside[-1][1]),(self.rightside[-1][0],self.rightside[-1][1]),(255,255,0))
            
        

        return image


    

if __name__ == '__main__':
    
    ##하나만 비교

    D2 = D1221()
    src = imread("03_connector_2-01/본딩 - 넘침/1. 원본.bmp")
    src = FitToWindowSize(src)
    dst = src.copy()
    output = D2.drawconT(src, dst)
    cv2.imshow('dst', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    # # 여러개 한꺼번에 비교
    # for i in range(13):
    #     print(i+1)
    #     src = imread("./03_connecter/"+str(i+1)+".bmp")
    #     D2 = D1221()
    #     src = FitToWindowSize(src)
    #     dst = src.copy()
    #     output = D2.drawconT(src, dst)
    #     cv2.imshow('dst', output)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        