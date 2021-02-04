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
filename = 'tomato.jpg'
src = imread(filename,cv2.IMREAD_COLOR)
if src is None:
    print("이미지를 읽을 수 없습니다.")
    exit(1)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

#트렉바 윈도우 생성
cv2.namedWindow("Track Bar", cv2.WINDOW_NORMAL) 

cv2.createTrackbar("h-Min", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Min", "Track Bar", 0)
cv2.createTrackbar("h-Max", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Max", "Track Bar", 180)

cv2.createTrackbar("h-Min2", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Min2", "Track Bar", 0)
cv2.createTrackbar("h-Max2", "Track Bar", 0,180, onChange)
cv2.setTrackbarPos("h-Max2", "Track Bar", 180)

cv2.createTrackbar("s-Min", "Track Bar", 0,255, onChange)
cv2.setTrackbarPos("s-Min", "Track Bar", 0)
cv2.createTrackbar("s-Max", "Track Bar", 0,255, onChange)
cv2.setTrackbarPos("s-Max", "Track Bar", 255)

cv2.createTrackbar("v-Min", "Track Bar", 0,255, onChange)
cv2.setTrackbarPos("v-Min", "Track Bar", 0)
cv2.createTrackbar("v-Max", "Track Bar", 0,255, onChange)
cv2.setTrackbarPos("v-Max", "Track Bar", 255)


ilowH = 0
ihighH = 180

ilowS=0
ihighS=255

ilowV=0
ihighV=255

while cv2.waitKey(1) != ord('q'):
    ilowH = cv2.getTrackbarPos("h-Min","Track Bar")
    ihighH = cv2.getTrackbarPos("h-Max","Track Bar")

    ilowH2 = cv2.getTrackbarPos("h-Min2","Track Bar")
    ihighH2 = cv2.getTrackbarPos("h-Max2","Track Bar")

    ilowS = cv2.getTrackbarPos("s-Min","Track Bar")
    ihighS = cv2.getTrackbarPos("s-Max","Track Bar")

    ilowV = cv2.getTrackbarPos("v-Min","Track Bar")
    ihighV = cv2.getTrackbarPos("v-Max","Track Bar")

    lower_blue = (ilowH, ilowS, ilowV)
    upper_blue = (ihighH, ihighS, ihighV)
    
    lower_blue2 = (ilowH2, ilowS, ilowV)
    upper_blue2 = (ihighH2, ihighS, ihighV)

    ## Apply the cv2.inrange method to create a mask
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask = cv2.bitwise_or(mask1, mask2)
    image2 = src.copy()
    ## Apply the mask on the image to extract the original color
    image2 = cv2.bitwise_and(image2, image2, mask=mask)
    cv2.imshow("Track Bar", image2)


cv2.destroyAllWindows()


