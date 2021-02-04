import cv2
import numpy as np    
from win32api import GetSystemMetrics

# opencv 한글 경로 지원 하지 않음
# numpy를 통해 우회하는 경로로 파일 로드
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
    # from win32api import GetSystemMetrics
    # print("Width =", GetSystemMetrics(0))
    # print("Height =", GetSystemMetrics(1))
    #이미지 크기 얻기
    print('image {}'.format(image.shape))
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
    


# cv2.imread가 아님/ 위에 정의한 imread()함수 사용
filename = 'chess.jpg'
# filename = 'images/sausage.jpg'
#filename = 'E:/OneDrive/Jupyter/opencv/PCB2/양품/양품1_2.bmp'
image = imread(filename,cv2.IMREAD_COLOR)
if image is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)

#image split -> BGR to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(hsv)

# mask = cv2.inRange(h, 100,107)
# blue = cv2.bitwise_and(hsv, hsv, mask=mask)
# blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)

# convert color to hsv because it is easy to track colors in this color model

ilowH=98; ilowS=106; ilowV=106
ihighH=110 ; ihighS=255; ihighV=255

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_blue = (ilowH, ilowS, ilowV)
upper_blue = (ihighH, ihighS, ihighV)
# Apply the cv2.inrange method to create a mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Apply the mask on the image to extract the original color
image = cv2.bitwise_and(image, image, mask=mask)


# rgb <--> hsv : http://colorizer.org/
resized_image = FitToWindowSize(image)
cv2.imshow("image",resized_image)
cv2.waitKey(0)
cv2.destoryAllWindows()