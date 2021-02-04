import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QSlider
from win32api import GetSystemMetrics

import cv2
import numpy as np

class MyApp(QtWidgets.QDialog):

    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi('20-12-15qt_이미지변경3.ui',self)
        self.count = 0
        self.filename=''
        self.threshold_option = 0

        #버튼
        self.loadBtn = self.findChild(QtWidgets.QPushButton,'loadBtn')
        self.loadBtn.clicked.connect(self.loadBtnClicked)
        self.procBtn = self.findChild(QtWidgets.QPushButton,'procBtn')
        self.procBtn.clicked.connect(self.procBtnClicked)

        self.src = self.findChild(QtWidgets.QLabel,'imgSrc')     
        # self.src.setPixmap(QtGui.QPixmap("chess.jpg"))
        # self.src.setScaledContents(True)
        # self.img = self.imread("chess.jpg")
        
        self.dst = self.findChild(QtWidgets.QLabel,'imgDst')
        # self.dst.clicked.connect(self.dstClicked)

        self.filePath = self.findChild(QtWidgets.QLineEdit,'filePath')
        self.filePath.clear()

        self.thr_value = self.findChild(QtWidgets.QLabel, 'label_threshold')

        self.hSlider = self.findChild(QtWidgets.QSlider, 'hSlider')
        self.hSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSlider.setMinimum(0)
        self.hSlider.setMaximum(255)
        self.hSlider.setTickInterval(10)
        self.hSlider.valueChanged.connect(self.hSliderChanged)

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
   
    
    # 이미지 로드(Image load)
    def loadBtnClicked(self):
        path = './PCB2/신규 pcb 이미지'
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
            
            img_gray = self.imgThr

            outImg = self.processingImage(img_rgb, img_gray)
            self.displayOutputImage(outImg,'dst')
        else:
            self.filename = "파일을 로드 하세요"
            self.filePath.setText(self.filename)

    # 결과이미지 출력
    def displayOutputImage(self, outImage, position):
        img_info = outImage.shape
        if outImage.ndim == 2 : # 그래이 영상
            qImg = QtGui.QImage(outImage, img_info[1], img_info[0], 
                            img_info[1]*1, QtGui.QImage.Format_Grayscale8)
        else :
            qImg = QtGui.QImage(outImage, img_info[1], img_info[0], 
                            img_info[1]*img_info[2], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
            
        if position == 'src' :
            self.src.setPixmap(pixmap)
            self.src.setScaledContents(True)
        else :
            self.dst.setPixmap(pixmap)
            self.dst.setScaledContents(True)


    #한글 포함된 이미지 파일 읽어오기
    def imread(self, filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
        try:
            n = np.fromfile(filename, dtype)
            img = cv2.imdecode(n, flags)
            return img
        except Exception as e:
            print(e)
            return None

    def processingImage(self, img_rgb, img_gray):
        #여기에 여러분이 작성할 코드를 넣으시면 됩니다.
        outImg = self.img.copy()

        if self.imgThr.ndim == 2 : # 그래이 영상
            pass
        else :
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
            ret1, img_gray = cv2.threshold(img_gray , 170, 255, 0)
            
            
        ## 모폴로지 변환
        # kernelsize = cv2.getStructuringElement(cv2.MORPH_CROSS(5, 5))
        # binary = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelsize, iterations=1)
        ## 오픈(침식연산(erosion)-> 확장연산(dilation)) 검은색 늘어남 / 클로즈(확장연산-> 침식연산) 흰색이 늘어남
        conts, h = cv2.findContours(img_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for i in range(len(conts)):
            area = cv2.contourArea(conts[i])
            if area > 100:
                cv2.drawContours(outImg, conts[i], -1, (0, 255, 0), 3)              
                x, y, w, h = cv2.boundingRect(conts[i])
                # column = contours[i][0][0][0] #가로 좌표(x)
                # row = contours[i][0][0][1]    #세로 좌표(y)
                #cv2.rectangle(outImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(outImg, str(area), (x-20, y-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)
                #cv2.putText(outImg,str(area),(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
        
        img = self.FitToWindowSize(outImg)
        cv2.imshow("img", img)
        outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)
        return outImg

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