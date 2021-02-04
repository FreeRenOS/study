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
    


# cv2.imread가 아님/ 위에 정의한 imread()함수 사용
filename = 'sausage.jpg'
image = imread(filename,cv2.IMREAD_COLOR)
if image is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)

#opencv bgr
b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]

height, width = image.shape[:2]
zero = np.zeros((height,width,1), dtype=np.uint8)

yellow = cv2.merge((zero,g,r))
cyan = cv2.merge((b,g,zero))
magenta = cv2.merge((b,zero,r))


resized_image = FitToWindowSize(image)
resized_b = FitToWindowSize(b)
resized_g = FitToWindowSize(g)
resized_r = FitToWindowSize(r)

yellow_r = FitToWindowSize(yellow)
cyan_r = FitToWindowSize(cyan)
magenta_r = FitToWindowSize(magenta)

cv2.imshow("image",resized_image)
# cv2.imshow("b",resized_b)
# cv2.imshow("g",resized_g)
# cv2.imshow("r",resized_r)
cv2.imshow("yellow",yellow_r)
cv2.imshow("cyan",cyan_r)
cv2.imshow("magenta",magenta_r)

cv2.waitKey(0)
cv2.destoryAllWindows()