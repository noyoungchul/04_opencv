import cv2
import numpy as np
import datetime
import os

Car = "Lincense Plate Extractor"
img = cv2.imread("../img/car05.jpg")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

# 출력 이미지 크기
width = 300
height = 150
pts2 = np.float32([[0,0], [width-1,0], 
                   [width-1,height-1], [0,height-1]])

# 저장 디렉토리 생성
save_dir = '../extracted_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def onMouse(event, x, y, flags, param):
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x,y), 10, (0,255,0), -1)
        cv2.imshow(Car, draw)

        pts[pts_cnt] = [x,y]
        pts_cnt += 1
        # 4개 좌표 정렬
        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis =1)

            topLeft = pts[np.argmin(sm)]         
            bottomRight = pts[np.argmax(sm)]     
            topRight = pts[np.argmin(diff)]     
            bottomLeft = pts[np.argmax(diff)]  

            # 변환전 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

            
            width = 300
            height = 150
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])

            # 원근 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
            cv2.imshow('Extracted Plate', result)

            # 파일 저장
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/plate_{timestamp}.png"
            success = cv2.imwrite(filename, result)

            if success:
                print(f" 번호판 저장 완료: {filename}")
                cv2.imshow('Extracted_Plated', result)
            else:
                print(" 저장 실패!")


cv2.imshow("License Plate Extractor", draw)
cv2.setMouseCallback("License Plate Extractor", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()