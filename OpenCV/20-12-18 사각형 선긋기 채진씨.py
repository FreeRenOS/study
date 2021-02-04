def processingImage(img_rgb):
    outImg = img_rgb.copy()
    img_gray = cv2.cvtColor(outImg, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(img_binary)
    blur = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
    edges = cv2.Canny(blur, 55, 452, apertureSize = 3)
    
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # 가장 긴 countor 반환
    cnt_max = max(contours, key=lambda c: cv2.arcLength(c, True))
    # cv2.drawContours(outImg, [cnt_max], 0, (0,0,0), 4) #확인용

    # 가장 왼쪽 값, 오른쪽 값으로 부터 20픽셀 안으로 떨어진 애들만 추려냄
    cnt = cnt_max.reshape(-1,2)
    maxXY = np.amax(cnt, axis=0)
    minXY = np.amin(cnt, axis=0)
    print(minXY,maxXY)
    cnt = cnt[np.where((cnt[:,0] <= minXY[0]+20)| (cnt[:,0] >= maxXY[0]-20))]
    # cv2.polylines(outImg, [cnt], True, (0, 255, 255), 10) #확인용

    # 그 애들로 최소 사각형 잡고 중심점 잡음
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    center = np.mean(box, axis=0, dtype=np.int0)
    box = np.int0(box)
    cv2.circle(outImg, tuple(center), 10, (255,255,0), -1)
    cv2.drawContours(outImg, [box], 0, (0,0,255), 2)
    

    # 우리가 원하는 내부 사각형을 잡기위해 다시 컨투어스 잡음
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours)) #확인용
    
    # 컨투어의 무게중심이 위에 구한 중심점과 일정 거리 이상이고, 컨투어영역이 일정 이상일 때
    center_list = []
    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        lenth = cv2.arcLength(cnt, True)
        center_C = (int(M["m10"] / (M["m00"]+1e-5)), int(M["m01"] / (M["m00"]+1e-5)))
        distance = center_C[1] - center[1]
        if area > 200 and area < 700:
            if distance < 70 and distance > -70:
                cv2.circle(outImg, center_C, 5, (225, 0, 0), -1)
                cv2.drawContours(outImg, [cnt], 0, (0,255,255), 5)
                center_list.append(center_C)
    # print("center_list",len(center_list)) #확인용

    # x좌표 기준으로 오름차순 정렬
    center_arr = np.array(center_list)
    center_arr = center_arr[center_arr[:,0].argsort()]
    
    # 중심점보다 위에 있는지 아래있는지에 따라 숫자 부여
    i = 0; j=0
    for c in center_arr:
        if c[1] < center[1]:
            i += 1
            cv2.putText(outImg, str(i), (c[0]-20,c[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        else:
            j += 1
            cv2.putText(outImg, str(j), (c[0]-20,c[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

    return outImg