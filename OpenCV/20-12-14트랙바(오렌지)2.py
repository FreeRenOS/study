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

def onChange(pos):
    pass        

# cv2.imread가 아님/ 위에 정의한 imread()함수 사용
filename = 'orange.png'
src = imread(filename,cv2.IMREAD_COLOR)
if src is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)


#트렉바 윈도우 생성
cv2.namedWindow("Track Bar")

cv2.createTrackbar("h-Min", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Min", "Track Bar", 0)
cv2.createTrackbar("h-Max", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Max", "Track Bar", 180)

ilowS=0; ilowV=0
ihighS=255; ihighV=255

while cv2.waitKey(1) != ord('q'):
    ilowH = cv2.getTrackbarPos("h-Min","Track Bar")
    ihighH = cv2.getTrackbarPos("h-Max","Track Bar")
    lower_blue = (ilowH, ilowS, ilowV)
    upper_blue = (ihighH, ihighS, ihighV)
    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Apply the mask on the image to extract the original color
    dst = src.copy()
    dst = cv2.bitwise_and(dst, dst, mask=mask)
    cv2.imshow("Track Bar",dst)

cv2.destroyAllWindows()