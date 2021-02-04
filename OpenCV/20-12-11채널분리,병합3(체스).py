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

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img, w, h, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('img', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                cv2.imshow('img', img_draw)
                roi = img[y0:y0+h, x0:x0+w]
                cv2.imshow('cropped', roi)
                cv2.moveWindow('cropped', 0, 0)
                #cv2.imwrite('./cropped.png', roi)
            else:
                cv2.imshow('img', img)
                print('drag should start from left-top side')

isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255,0,0),(0,0,255)

# cv2.imread가 아님/ 위에 정의한 imread()함수 사용
filename = './images/chess.jpg'
image = imread(filename,cv2.IMREAD_COLOR)
if image is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)

img = image.copy()
img = FitToWindowSize(image)
img2 = FitToWindowSize(image)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#roi = dst[50:550, 150:650]
# roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# roi2 = np.stack((roi2,)*3, -1)
#roi2 = cv2.merge((roi2,roi2,roi2)) # 위 코드랑 같은 결과

# height, width = roi2.shape[:2]
# zero = np.zeros((height,width,1), dtype=np.uint8)

# img[y0:y0+h, x0:x0+w] = roi2



#opencv bgr
# b = image[:,:,0]
# g = image[:,:,1]
# r = image[:,:,2]

# height, width = image.shape[:2]
# zero = np.zeros((height,width,1), dtype=np.uint8)

# yellow = cv2.merge((zero,g,r))
# cyan = cv2.merge((b,g,zero))
# magenta = cv2.merge((b,zero,r))

# resized_image = FitToWindowSize(image)
# resized_b = FitToWindowSize(b)
# resized_g = FitToWindowSize(g)
# resized_r = FitToWindowSize(r)

# yellow_r = FitToWindowSize(yellow)
# cyan_r = FitToWindowSize(cyan)
# magenta_r = FitToWindowSize(magenta)

# cv2.imshow("image",resized_image)
# cv2.imshow("b",resized_b)
# cv2.imshow("g",resized_g)
# cv2.imshow("r",resized_r)

# cv2.imshow("yellow",yellow_r)
# cv2.imshow("cyan",cyan_r)
# cv2.imshow("magenta",magenta_r)

#cv2.imshow("image",image)
# cv2.imshow('dst',dst)

# cv2.imshow('img',img)
# cv2.setMouseCallback('img', onMouse)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 범위를 정하여 hsv이미지에서 원하는 색 영역을 바이너리 이미지로 생성한다
lowerBound = np.array([100, 105, 100])
upperBound = np.array([110, 255, 255])

# 앞서 선언한 범위값을 사용하여 바이너리 이미지를 얻는다.
mask = cv2.inRange(hsv, lowerBound, upperBound)

# mopology
kernelsize = np.ones((5, 5))
maskFinal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelsize)

# 원본이미지에서 범위값에 해당하는 영상부분을 흭득한다.
img2 = cv2.bitwise_and(img2, img2, mask=maskFinal)

# roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# contours 를 찾기위한 이진화
ret1, thr = cv2.threshold(maskFinal , 127, 255, 0)

conts, h = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
# cv2.drawContours(thr, conts, -1, (0, 255, 0), 3)

for i in range(len(conts)):
    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(conts[i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


# ret, binary = cv2.threshold(roi2, 110, 255, cv2.THRESH_BINARY)
# binary = cv2.bitwise_not(binary)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for i in range(len(contours)):
#     cv2.drawContours(roi2, [contours[i]], 0, (0, 0, 255), 2)
#     #cv2.putText(roi2, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
#     print(i, hierarchy[0][i])

# cv2.imshow("roi", roi2)
# cv2.imshow("binary", binary)
cv2.imshow("img", img)
cv2.imshow("image", img2)
# cv2.imshow("hsv", hsv)
cv2.imshow("thr", thr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# roi2 = np.stack((roi2,)*3, -1)
# height, width = roi2.shape[:2]

# img[y0:y0+h, x0:x0+w] = roi2

# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()


