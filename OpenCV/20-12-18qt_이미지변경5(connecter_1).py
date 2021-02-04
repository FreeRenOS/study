from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QSlider
import sys
import numpy as np
import pandas as pd
import cv2
from win32api import GetSystemMetrics

class MyApp(QtWidgets.QDialog):

    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi('20-12-16qt_이미지변경4.ui', self)
        self.count = 0
        self.lower = (0, 0, 0)
        self.upper = (180, 255, 255)
        self.lower2 = (0, 0, 0)
        self.upper2 = (180, 255, 255)

        self.filename=''
        self.threshold_option = 0
        
        # 버튼
        self.loadBtn = self.findChild(QtWidgets.QPushButton,'loadBtn')
        self.loadBtn.clicked.connect(self.loadBtnClicked)
        self.procBtn = self.findChild(QtWidgets.QPushButton,'procBtn')
        self.procBtn.clicked.connect(self.procBtnClicked)
        self.CircleBtn = self.findChild(QtWidgets.QPushButton, 'CircleBtn')
        self.CircleBtn.clicked.connect(self.Connect_Detect)


        ## QLineEdit
        self.filePath = self.findChild(QtWidgets.QLineEdit, 'filePath')
        self.filePath.clear()

        ## QLabel : 이미지창
        self.src = self.findChild(QtWidgets.QLabel, 'imgSrc')
        self.src.setPixmap(QtGui.QPixmap("./images/chess.jpg"))
        self.src.setScaledContents(True)
        self.img = self.imread("./images/chess.jpg")
        self.dst = self.findChild(QtWidgets.QLabel, 'imgDst')

        ## QLabel
        self.thr_value = self.findChild(QtWidgets.QLabel, 'label_threshold')
        self.hSlider = self.findChild(QtWidgets.QSlider, 'hSlider')
        self.hSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider.valueChanged.connect(self.hSliderChanged)

        self.thr_value_H = self.findChild(QtWidgets.QLabel, 'label_threshold_H')
        self.hSlider_H = self.findChild(QtWidgets.QSlider, 'hSlider_H')
        self.hSlider_H.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_H.valueChanged.connect(self.hSlider_HChanged)

        self.thr_value_H2 = self.findChild(QtWidgets.QLabel, 'label_threshold_H2')
        self.hSlider_H2 = self.findChild(QtWidgets.QSlider, 'hSlider_H2')
        self.hSlider_H2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_H2.valueChanged.connect(self.hSlider_H2Changed)

        self.thr_value_H3 = self.findChild(QtWidgets.QLabel, 'label_threshold_H3')
        self.hSlider_H3 = self.findChild(QtWidgets.QSlider, 'hSlider_H3')
        self.hSlider_H3.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_H3.valueChanged.connect(self.hSlider_H3Changed)

        self.thr_value_H4 = self.findChild(QtWidgets.QLabel, 'label_threshold_H4')
        self.hSlider_H4 = self.findChild(QtWidgets.QSlider, 'hSlider_H4')
        self.hSlider_H4.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_H4.valueChanged.connect(self.hSlider_H4Changed)

        self.thr_value_S = self.findChild(QtWidgets.QLabel, 'label_threshold_S')
        self.hSlider_S = self.findChild(QtWidgets.QSlider, 'hSlider_S')
        self.hSlider_S.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_S.valueChanged.connect(self.hSlider_SChanged)

        self.thr_value_S2 = self.findChild(QtWidgets.QLabel, 'label_threshold_S2')
        self.hSlider_S2 = self.findChild(QtWidgets.QSlider, 'hSlider_S2')
        self.hSlider_S2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_S2.valueChanged.connect(self.hSlider_S2Changed)

        self.thr_value_V = self.findChild(QtWidgets.QLabel, 'label_threshold_V')
        self.hSlider_V = self.findChild(QtWidgets.QSlider, 'hSlider_V')
        self.hSlider_V.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_V.valueChanged.connect(self.hSlider_VChanged)

        self.thr_value_V2 = self.findChild(QtWidgets.QLabel, 'label_threshold_V2')
        self.hSlider_V2 = self.findChild(QtWidgets.QSlider, 'hSlider_V2')
        self.hSlider_V2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider_V2.valueChanged.connect(self.hSlider_V2Changed)

        #QComboBox
        self.comboThr = self.findChild(QtWidgets.QComboBox, 'comboThr')
                
        # # 이진화 옵션번호가 0~5 OTSU가 8 TRIANGLE이16인 이유는 여러개를 같이 쓸수 있으므로.
        # # 숫자를 더해서 같은 값이 나오면 안된다.
        self.thrList = ["BINARY",
                    "BINARY_INV",
                    "TRUNC",
                    "TOZERO",
                    "TOZERO_INV",
                    "MASK",
                    "OTSU",
                    "TRIANGLE",
                    "ADAPTIVE_MEAN",
                    "ADAPTIVE_GAUSSIAN"]
        
        self.modelist = [cv2.THRESH_BINARY,
                         cv2.THRESH_BINARY_INV,
                         cv2.THRESH_TRUNC,
                         cv2.THRESH_TOZERO,
                         cv2.THRESH_TOZERO_INV,
                         cv2.THRESH_MASK,
                         cv2.THRESH_OTSU,
                         cv2.THRESH_TRIANGLE,
                         cv2.ADAPTIVE_THRESH_MEAN_C,
                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C]

        self.comboThr.addItems(self.thrList) 
        self.comboThr.activated[str].connect(self.ComboBoxEvent) 
        self.show()

    def ComboBoxEvent(self, text):
        index = self.thrList.index(text)
        self.threshold_option = self.modelist[index]
        self.hSliderChanged()

    def hSliderChanged(self):
        if self.filename !='':
            self.thr_value.setText("임계값 :"+str(self.hSlider.value()))
            img_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(img_gray, self.hSlider.value(), 255, self.threshold_option)
            self.imgThr = binary
            self.displayOutputImage(binary, 'dst')
        else :
            self.filePath.setText("예외발생 : 파일을 선택해 주세요!!!!")         

    def hSlider_HChanged(self):
        self.thr_value_H.setText("H Min값 :"+str(self.hSlider_H.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.lower = (self.hSlider_H.value(), self.hSlider_S.value(), self.hSlider_V.value())
        self.upper = (self.hSlider_H2.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        self.lower2 = (self.hSlider_H3.value(), self.hSlider_S.value(), self.hSlider_V.value())
        self.upper2 = (self.hSlider_H4.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_H2Changed(self):
        self.thr_value_H2.setText("H Max값 :"+str(self.hSlider_H2.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        self.upper = (self.hSlider_H2.value(), self.hSlider_S2.value(), self.hSlider_V2.value())

        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_H3Changed(self):
        self.thr_value_H3.setText("H Min값_2 :"+str(self.hSlider_H3.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        self.lower2 = (self.hSlider_H3.value(), self.hSlider_S.value(), self.hSlider_V.value())

        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_H4Changed(self):
        self.thr_value_H4.setText("H Max값_2 :"+str(self.hSlider_H4.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        self.upper2 = (self.hSlider_H4.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_SChanged(self):
        self.thr_value_S.setText("S Min값 :"+str(self.hSlider_S.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.lower = (self.hSlider_H.value(), self.hSlider_S.value(), self.hSlider_V.value())
        
        self.lower2 = (self.hSlider_H3.value(), self.hSlider_S.value(), self.hSlider_V.value())
        
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_S2Changed(self):
        self.thr_value_S2.setText("S Max값 :"+str(self.hSlider_S2.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        self.upper = (self.hSlider_H2.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        
        self.upper2 = (self.hSlider_H4.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_VChanged(self):
        self.thr_value_V.setText("V Min값 :"+str(self.hSlider_V.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.lower = (self.hSlider_H.value(), self.hSlider_S.value(), self.hSlider_V.value())
 
        self.lower2 = (self.hSlider_H3.value(), self.hSlider_S.value(), self.hSlider_V.value())
 
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    def hSlider_V2Changed(self):
        self.thr_value_V2.setText("V Max값 :"+str(self.hSlider_V2.value()))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
     
        self.upper = (self.hSlider_H2.value(), self.hSlider_S2.value(), self.hSlider_V2.value())

        self.upper2 = (self.hSlider_H4.value(), self.hSlider_S2.value(), self.hSlider_V2.value())
        mask1 = cv2.inRange(hsv, self.lower, self.upper)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        imghsv = self.img.copy()
        imghsv = cv2.bitwise_and(imghsv, imghsv, mask=mask)
        self.img2 = imghsv.copy()
        imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2RGB)
        self.displayOutputImage(imghsv, 'dst')

    ## 이미지 로드(Image load)
    def loadBtnClicked(self):
        path = './01_connecter_1'
        filter = "All Images(*.jpg; *.png; *.bmp);;JPG (*.jpg);;PNG(*.png);;BMP(*.bmp)"
        fname = QFileDialog.getOpenFileName(self, "파일 불러오기", path, filter)
        self.filename = str(fname[0])
        if fname[0] !='' : # 파일로드시 파일 선택하지 않고 취소할 때 예외처리
            filename = str(fname[0])
            self.filePath.setText(filename)
            self.img = self.imread(filename) #cv2.imread가 한글경로를 지원하지 않음
            self.img2 = self.img.copy()
            self.imgThr = self.img.copy()
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.displayOutputImage(img_rgb, 'src') # 'src'(왼쪽) or 'dst'(오른쪽) 창
            self.count += 1
        else :
            self.filePath.setText("예외발생 : 파일을 선택해 주세요!!!!") 


    def procBtnClicked(self):
        if self.count != 0:
            # img = self.imread(self.filename) #opencv는 한글 지원 안함
            img_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
            # img_bgr = self.img.copy()
            
            img_gray = self.imgThr.copy()

            outImg = self.processingImage(img_rgb, img_gray)
            self.displayOutputImage(outImg,'dst')
        else:
            self.filename = "파일을 로드 하세요"
            self.filePath.setText(self.filename)


    def CircleBtnClicked(self):
        if self.count != 0:
            img = self.img.copy()
            imgcircle = self.img.copy()

            if imgcircle.ndim == 2 : # 그래이 영상
                pass
            else :
                imgcircle = cv2.cvtColor(imgcircle, cv2.COLOR_BGR2GRAY)
           
            _, imgcircle = cv2.threshold(imgcircle, 90, 255, type = cv2.THRESH_BINARY_INV) # 이진화
            imgcircle = cv2.medianBlur(imgcircle,5)
            circles = cv2.HoughCircles(imgcircle, cv2.HOUGH_GRADIENT, 1, 1800, param1 = 150, param2 = 5, minRadius = 400, maxRadius = 440)
            
            circles =np.uint16(np.around(circles))

            for i in circles[0]:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 3)

            img = self.FitToWindowSize(img) 

            cv2.imshow("img", img)
            outImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.displayOutputImage(outImg,'dst')

        else:
            self.filename = "파일을 로드 하세요"
            self.filePath.setText(self.filename) 

    ## 한글 포함된 이미지 파일 읽어오기
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        try :
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            return img

        except Exception as e:
            print(e)
            return None


    ## 결과이미지 출력
    def displayOutputImage(self, outImg, position):
        img_info = outImg.shape
        if outImg.ndim == 2: ## 그레이 영상
            qImg = QtGui.QImage(outImg, img_info[1], img_info[0],
                                    img_info[1]*1, QtGui.QImage.Format_Grayscale8)
        else:
            qImg = QtGui.QImage(outImg, img_info[1], img_info[0],
                                    img_info[1]*img_info[2], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        if position == 'src' :
            self.src.setPixmap(pixmap)
            self.src.setScaledContents(True)
        else :
            self.dst.setPixmap(pixmap)
            self.dst.setScaledContents(True)

    def processingImage(self, img_rgb, img_gray):
        #여기에 여러분이 작성할 코드를 넣으시면 됩니다.
        outImg = self.img.copy()

        if self.imgThr.ndim == 2 : # 그래이 영상
            pass
        else :
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
            ret1, img_gray = cv2.threshold(img_gray, self.hSlider.value(), 255, self.threshold_option)

        ## 모폴로지 변환
        kernelsize = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernelsize, iterations=1)
        ## 오픈(침식연산(erosion)-> 확장연산(dilation)) 검은색 늘어남 / 클로즈(확장연산-> 침식연산) 흰색이 늘어남

        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        ## 가장 큰 크기 찾아서 그리기
        # c = max(contours, key = cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(outImg, (x, y), (x + w, y + h), (255, 0, 255), 2)

        ## x의 min과 max 찾기
        ## y의 min과 max 찾기
        x_min, x_max = 0,0
        y_min, y_max = 0,0
        x_value = list()
        y_value = list()
    
        ##찾은 Contour개수만큼 반복하여 그리기
        for i in range(len(contours)):

            area = cv2.contourArea(contours[i])
            
            if area > 200:
                ## 모멘트 구하기
                M = cv2.moments(contours[i])
                cX = int(M['m10'] / M['m00']+1e-5)
                cY = int(M['m01'] / M['m00']+1e-5)
                cv2.circle(outImg, (cX, cY), 3, (0, 0, 255), -1)

                ## Contour 그리기
                x, y, w, h = cv2.boundingRect(contours[i])
                cv2.drawContours(outImg, contours[i], -1, (0, 255, 0), 3)    

                x_value.append(x) ## x의 min과 max 찾기
                x_value.append(x+w) ## x의 min과 max 찾기
                y_value.append(y) ## y의 min과 max 찾기
                y_value.append(y+h) ##  x의 min과 max 찾기 


                
                ## 사각형 그리기
                cv2.rectangle(outImg, (x, y), (x + w, y + h), (255, 0, 255), 2)
                # cv2.rectangle(outImg, (x, y), (x + w, y + h), (255, 0, 255), 2)

                ## 최소한의 사각형 표시
                # rect = cv2.minAreaRect(contours[i])
                # box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
                # box = np.int0(box)          # 정수로 변환
                # cv2.drawContours(outImg, [box], -1, (255,255,0), 3)

                
                ## Contour 면적 화면에 표시하기(putText)
                column = contours[i][0][0][0] #가로 좌표(x)
                row = contours[i][0][0][1]    #세로 좌표(y)
                
                # cv2.putText(outImg, str(area), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
                cv2.putText(outImg, str(h), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                # cv2.putText(outImg, str(area), (column-20, row+20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)
        
        x_min = min(x_value)
        x_max = max(x_value)
        y_min = min(y_value)
        y_max = max(y_value)

        print(x_min)
        print(x_max)
        print(y_min)
        print(y_max)

        ##최대,최소값으로 네모그리기
        cv2.rectangle(outImg, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        ##화면크기에 영상 맞추기
        img = self.FitToWindowSize(outImg) 
        cv2.imshow("img", img)
        outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)
        return outImg

    def Connect_Detect(self):
        if self.count != 0:
            roi=[]
            
            outImg = self.img.copy()
            img_roi = self.img.copy()
            
            img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            ret1, img_roi = cv2.threshold(img_roi, 24, 255, cv2.THRESH_BINARY_INV)

            ##관심영역 짜르고 네모영역 그리기
            #검은색 영역 만들기
            RoiZero = np.zeros(img_roi.shape, dtype=np.uint8)
            
            # 이미지를 자르고 크기를 구한다.
            img_roi = img_roi[830:1140, 540:2055].copy()
            width, height = img_roi.shape
            print(width, height)

            # 자른 이미지를 소스 이미지에 붙인다.
            RoiZero[830:1140, 540:2055] = img_roi

            # 모폴로지 변환
            kernelsize = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_roi = cv2.morphologyEx(RoiZero, cv2.MORPH_CLOSE, kernelsize, iterations=1)

            
            # 컨투어 찾기
            contours, hierarchy = cv2.findContours(img_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            ##찾은 Contour개수만큼 반복하여 그리기
            for i in range(len(contours)):

                area = cv2.contourArea(contours[i])
                
                if area > 1000:
                    ## 윤곽선 둘러싸는 사각형 좌표 너비 높이 구하기
                    x, y, w, h = cv2.boundingRect(contours[i])
 
                    ## 직사각형 찾기(2개)
                    if ( (h > 140) & (h <= 180) ) :
                        roi.append(i)
 
            # 최소한의 사각형 표시
            # rect = cv2.minAreaRect(contours[roi[0]])
            # box = cv2.boxPoints(rect)   # 중심점과 각도를 4개의 꼭지점 좌표로 변환
            # box = np.int0(box)          # 정수로 변환
            # cv2.drawContours(outImg, [box], -1, (255,255,0), 3)

            ## 특정 직사각형 외각선 긋기
            x0, y0, w0, h0 = cv2.boundingRect(contours[roi[0]])
            x1, y1, w1, h1 = cv2.boundingRect(contours[roi[1]])

            if x0 < x1 :
                ## 0번 윤곽선이 왼쪽에 있을때
                cv2.line(outImg, (x0, y0), (x0, y0+h0), (0, 255, 255), 2)

                cv2.putText(outImg, str(x0), (x0, y0+int(h0/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(outImg, str(y0), (x0, y0+int(h0/2)+30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.line(outImg, (x1+w1, y1), (x1+w1, y1+h1), (0, 255, 255), 2)
                cv2.putText(outImg, str(y1+h1), (x1+w1, y1+int(h1/2)+30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(outImg, str(x1+w1), (x1+w1, y1+int(h1/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                
            else :
                ## 1번 윤곽선이 왼쪽에 있을때
                cv2.line(outImg, (x1, y1), (x1, y1+h1), (255, 0, 255), 2)
                cv2.putText(outImg, str(x1), (x1, y1+int(h1/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(outImg, str(y1), (x1, y1+int(h1/2)+30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                
                cv2.line(outImg, (x0+w0, y0), (x0+w0, y0+h0), (255, 0, 255), 2)
                cv2.putText(outImg, str(y0+h0), (x0+w0, y0+int(h0/2)+30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(outImg, str(x0+w0), (x0+w0, y0+int(h0/2)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                
            ################################################################################################################################
            ## 커넥트 영역 지정하고 찾기 ##

            img_gray = self.imgThr.copy()
            if self.imgThr.ndim == 2 : # 그래이 영상
                pass
            else :
                img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
                ret1, img_gray = cv2.threshold(img_gray, self.hSlider.value(), 255, self.threshold_option)

            #검은색 영역 만들기
            RoiZero = np.zeros(img_roi.shape, dtype=np.uint8)
            rx=15
            ry=50
            if x0 < x1 :    
                # 이미지를 자르른다
                img_gray = img_gray[y0-ry:y1+h1+ry, x0-rx:x1+w1+rx].copy()
                # 자른 이미지를 소스 이미지에 붙인다.
                RoiZero[y0-ry:y1+h1+ry, x0-rx:x1+w1+rx] = img_gray
                cv2.rectangle(outImg, (x0-rx, y0-ry), (x1+w1+rx, y1+h1+ry), (255, 0, 255), 2)
            else:
                img_gray = img_gray[y1-ry:y0+h0+ry, x1-rx:x0+w0+rx].copy()
                # 자른 이미지를 소스 이미지에 붙인다.
                RoiZero[y1-ry:y0+h0+ry, x1-rx:x0+w0+rx] = img_gray
                cv2.rectangle(outImg, (x1-rx, y1-ry), (x0+w0+rx, y0+h0+ry), (255, 0, 255), 2)

            ## 모폴로지 변환
            kernelsize = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_gray = cv2.morphologyEx(RoiZero, cv2.MORPH_CLOSE, kernelsize, iterations=1)
            ## 오픈(침식연산(erosion)-> 확장연산(dilation)) 검은색 늘어남 / 클로즈(확장연산-> 침식연산) 흰색이 늘어남

            contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            MomentList=[]
            for i in range(len(contours)):

                    area = cv2.contourArea(contours[i])
                    
                    if area > 1000:
                        ## 모멘트 구하기
                        M = cv2.moments(contours[i])
                        cX = int(M['m10'] / M['m00']+1e-5)
                        cY = int(M['m01'] / M['m00']+1e-5)
                        cv2.circle(outImg, (cX, cY), 3, (0, 0, 255), -1)
                        MomentList.append([cX,cY])
                        ## Contour 그리기
                        x, y, w, h = cv2.boundingRect(contours[i])
                        cv2.drawContours(outImg, contours[i], -1, (0, 255, 0), 3)    


                        ## 사각형 그리기
                        # cv2.rectangle(outImg, (x, y), (x + w, y + h), (255, 0, 255), 2)

                        # cv2.putText(outImg, str(i), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                        # cv2.putText(outImg, str('x: {}'.format(x)), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                        # cv2.putText(outImg, str('y: {}'.format(y)), (cX-50, cY+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
                        # cv2.putText(outImg, str('x:{}\fy:{}'.format(x,y)), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
                        cv2.putText(outImg, str(area), (cX-50, cY-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            

            ################################################################################################################################
            
            img = self.FitToWindowSize(outImg) 
            cv2.imshow("img", img)
            outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)

            self.displayOutputImage(outImg,'dst')
        else:
            self.filename = "파일을 로드 하세요"
            self.filePath.setText(self.filename)

    # OPENCV 화면 출력 관련 함수 : 화면크기에 맞춰 이미지 출력
    def FitToWindowSize(self, image):
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

            
app = QtWidgets.QApplication(sys.argv)
window = MyApp()
app.exec_()