import cv2
import numpy as np
import function
import copy
import math

def equation(A, B): # 두점을 지나는 방정식의 기울기, 절편 구하기
    # A(A[0], [A[1]]), B(B[0], B[1])
    a = (B[1]-A[1]) / (B[0] - A[0])
    b = A[1] - a * A[0]

    return [a,b] # 기울기, 절편 return


def meet_point(a1, b1, a2, b2): # 두 직선의 기울기와 절편을 이용해 교점 구하기
    x = (b2-b1)/(a1-a2)
    y = a1*((b2-b1)/(a1-a2)) + b1
    return (int(x), int(y))

def point_to_point(x1, y1, x2, y2): # 두 점 사이의 거리
    x = abs(x2-x1)
    y = abs(y2-y1)
    length = math.sqrt((x*x) + (y*y))
    
    return length

def judge_circle_contour(circle_x, circle_y, radius, point_x, point_y): # 원방정식을 만족하는 점만 추출
    if (point_x - circle_x) ** 2 + (point_y - circle_y) ** 2 == radius ** 2:
        return 1
    else:
        return 0

def find_length_point(x,y,x1,y1):
    length = point_to_point(x,y,x1,y1)
    center_x = abs((x - x1))
    center_y = abs((y - y1))
    return length, (center_x, center_y)

# filename = []

# for i in range(30):
#     filename.append("./04_top/" + str(i) + ".bmp")

src = function.imread("./04_top/10.bmp")
src = function.FitToWindowSize(src) # 이미지 크기 조절 후에 이미지 처리한 상태
                                    # 이미지 처리 후 이미지 크기 조절해야
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# https://076923.github.io/posts/Python-opencv-14/
src_laplacian = cv2.Laplacian(src_gray, cv2.CV_8U, ksize= 3)    # 가장자리 검출
_, binary = cv2.threshold(src_laplacian, 80, 255, cv2.THRESH_BINARY)
#_, imageBinary = cv2.threshold(imageGray, 85, 255, type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
rect = binary.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel, iterations=2)


contours, hierarchy = cv2.findContours(rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 제일 큰 윤곽 구하기
max = max(contours, key= cv2.contourArea)
#cv2.drawContours(src, [max], 0, (0,255,0),2)

# 제일 큰 윤곽을 감싸는 최소 사각형 
min_rect = cv2.minAreaRect(max) # contour에 외접하면서 면적이 가장 작은 
box = cv2.boxPoints(min_rect) # minAreaRect로 얻은 직사각형 꼭지점 4개 좌표
box = np.int0(box)

cv2.drawContours(src, [box], 0 ,(0, 255, 255), 1)
target = ()
for i in range(box.shape[0]):
    target = (int(box[i][0]), int(box[i][1]))
    cv2.circle(src, target, 2, (0, 255, 255), 1)
    sentence = "(" + str(target[0]) + ", " + str(target[1]) + ")"
    cv2.putText(src, sentence, (int(box[i][0]), int(box[i][1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255) , 1)


# #코너검출
# corners = cv2.goodFeaturesToTrack(src_gray, 200, 0.01, 10, blockSize=10, useHarrisDetector=True, k=0.05)

# for i in corners:
#     cv2.circle(src, tuple(i[0]), 3, (0, 0, 255), 2)
#     #cv2.putText(src, str(i), tuple(i[0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

# 각 지점을 선으로 그어 교점 찾으려고
rec_point = []
for i in range(box.shape[0]):
    rec_point.append((int(box[i][0]), int(box[i][1])))

cv2.line(src, rec_point[0], rec_point[2], (0,255,255), 1)
cv2.line(src, rec_point[1], rec_point[3], (0,255,255), 1)

# 사각형 대각선 교점
center = meet_point(equation(rec_point[0], rec_point[2])[0], equation(rec_point[0], rec_point[2])[1], equation(rec_point[1], rec_point[3])[0], equation(rec_point[1], rec_point[3])[1])
cv2.circle(src, center, 3, (0, 255, 255) ,-1)
sentence = "Rect center (" + str(center[0]) + ", " + str(center[1]) + ")"
cv2.putText(src, sentence, center, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255) , 1)

# 원
_, binary = cv2.threshold(src_gray, 90, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
circle = binary.copy()


# 원 검출
large_circle = cv2.HoughCircles(circle, cv2.HOUGH_GRADIENT, 1, 100, param1 = 50, param2 = 100, minRadius = 50, maxRadius = 500)
small_circle = cv2.HoughCircles(circle, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 50, param2 = 30, minRadius = 50, maxRadius = 150)

for i in large_circle[0]:  # 큰 원
    cv2.circle(src, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 0), 1)
    sentence = "Large Circle center (" + str(int(i[0])) + ", " +str(int(i[1])) + ")"
    cv2.circle(src, (int(i[0]), int(i[1])), 3, (255, 255, 0) ,-1)
    cv2.putText(src, sentence, (int(i[0]), int(i[1])+35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0) , 1)

for i in small_circle[0]:  # 작은 원
    cv2.circle(src, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 0), 1)
    sentence = "Small Circle center (" + str(int(i[0])) + ", " +str(int(i[1])) + ")"
    cv2.circle(src, (int(i[0]), int(i[1])), 3, (255, 255, 0) ,-1)
    cv2.putText(src, sentence, (int(i[0]), int(i[1])+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0) , 1)

l_x, l_y, l_r = (int(large_circle[0][0][0]), int(large_circle[0][0][1]), int(large_circle[0][0][2]))
s_x, s_y, s_r = (int(small_circle[0][0][0]), int(small_circle[0][0][1]), int(small_circle[0][0][2]))
# cv2.imshow("2",circle[y-r: y+r, x-r: x+r])   # 좌표 슬라이스 할때는 반대로 들어가는 듯
# 이미지에 src[높이(행), 너비(열)]에서 잘라낼 영역을 설정
# https://076923.github.io/posts/Python-opencv-9/

circle[s_y-s_r-50: s_y+s_r+50, s_x-s_r-50: s_x+s_r+50] = 0
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
circle = cv2.morphologyEx(circle, cv2.MORPH_OPEN, kernel, iterations=1)

contours, hierarchy = cv2.findContours(circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#max = max(contours, key= cv2.contourArea)
#cv2.drawContours(src, [max], 0, (0,255,0),2)

# contour 좌표와 중심 좌표 길이가 가장 짧은 길이를 반지름으로 원 그림
length = []
for i in range(len(contours)):
    #cv2.drawContours(src, [contours[i]], 0, (0,255,0),2)
    p_len = point_to_point(s_x, s_y,contours[i][0][0][0], contours[i][0][0][1])
    length.append(p_len)
minlength_index = length.index(min(length))
minlength_val = length[minlength_index]

cv2.circle(src, (contours[minlength_index][0][0][0], contours[minlength_index][0][0][1]), 5, (0, 0, 255), -1)
cv2.circle(src, (s_x, s_y), int(minlength_val), (0, 0, 255), 1)

# 틀만 따오려고 mask 만듬
zeros_circle = np.zeros(circle.shape, dtype= "uint8")
mask = cv2.circle(zeros_circle, (l_x, l_y), l_r, (255,255,255), -1)
cv2.circle(mask, (s_x, s_y), s_r, (0, 0, 0), -1)
circle_cut = cv2.bitwise_and(circle, mask)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
circle_cut = cv2.morphologyEx(circle_cut, cv2.MORPH_OPEN, kernel, iterations=3)
circle_cut = cv2.morphologyEx(circle_cut, cv2.MORPH_CLOSE, kernel, iterations=3)

contours, hierarchy = cv2.findContours(circle_cut, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt_area = []
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    cnt_area.append(area)

cnt_area_sort = cnt_area.copy()
cnt_area_sort.sort(reverse=True)
second_area_index = cnt_area_sort[1] #2번째로 큰 윤곽 넓이

cv2.drawContours(src, [contours[cnt_area.index(second_area_index)]], 0, (0, 255, 0), 1)
target = contours[cnt_area.index(second_area_index)] # 넓이가 2번째로 큰 contour

center_target_length = []
for i in range(len(target)):
    x, y = target[i][0][0], target[i][0][1]
    length = point_to_point(center[0], center[1], x, y)
    center_target_length.append(int(length))

center_target_length_temp = copy.deepcopy(center_target_length)
center_target_length.sort()
median = (center_target_length[0] + center_target_length[-1])/2
median = int(median)

# print("center_target_length", center_target_length)
# print("center_target_length_temp", center_target_length_temp)

# contour 만드는 과정
cv2.circle(src, center, center_target_length[-1], (0,0,255), 1) # 센터와 컨투어 길이 제일 큰 값을 기준으로 원
cv2.circle(src, center, median, (0,0,255), 1)

cnt = np.zeros(circle.shape, dtype= "uint8")
cv2.drawContours(cnt, [target],  0, (255, 255, 255), 1)

cnt = cv2.dilate(cnt, kernel, iterations=2)
cnt = cv2.blur(cnt, (9,9))

mask = np.zeros(circle.shape, dtype="uint8")
cv2.circle(mask, center, median, (255, 255, 255), -1)
cv2.bitwise_not(mask)
cnt = cv2.bitwise_and(cnt, mask) # 만들어짐

# 코너 검출
corners = cv2.goodFeaturesToTrack(cnt, 20, 0.01, 30, blockSize=5, useHarrisDetector=True, k=0.03)
for i in corners:
    cv2.circle(src, tuple(i[0]), 3, (0, 255, 0), 2)

for i in range(len(corners)):
    point = (int(corners[i][0][0]), int(corners[i][0][1]))
    radius = median - int(minlength_val)
    cv2.line(src, center, point, (255, 255, 0), 1)
    cv2.circle(src, point, radius, (255, 255, 0), 1)

# 이미지 부드럽게 가우시안 블러링(Gaussian Blurring), 블러링 등을 사용해야

cv2.imshow("src",src)
cv2.imshow("contour", cnt)




cv2.waitKey(0)
cv2.destroyAllWindows()