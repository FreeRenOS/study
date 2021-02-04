# 파이썬은 한글 지원하지 않음
# numpy 이용해서 읽어 들임
import cv2
import numpy as np

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def img_resize(image) : # 컴퓨터 해상도에 맞춰서 크기 바꿈
    #윈도우 크기 얻기
    from win32api import GetSystemMetrics
    print("Width =", GetSystemMetrics(0))
    print("Height =", GetSystemMetrics(1))

    #이미지 크기 얻기
    print('image {}'.format(image.shape))
    win_w=GetSystemMetrics(0)
    win_h=GetSystemMetrics(1)
    img_h, img_w = image.shape[:2]

    if(img_h > win_h or img_w > win_w):   
        rate_width =  (win_w / img_w)
        rate_height =  (win_h / img_h)
        scale = rate_width if (rate_width < rate_height) else rate_height

    return cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# cv2.imread가 아님/ 위에 정의한 imread()함수

image = imread('sausage.jpg')

if image is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)


b, g, r = cv2.split(image)
inversebgr = cv2.merge((r, g, b))

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("inverse", inversebgr)


    
b2=img_resize(b)
cv2.imshow("b2", b2)
#cv2.imshow('image_resize',image_resized)

# cv2.namedWindow('image')
# cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destoryAllWindows()