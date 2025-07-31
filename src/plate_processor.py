import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


#     1단계: 이미지 로드    
def load_plate_image(filename):
    path = f'../extracted_plates/{filename}.png'
    if not os.path.exists(path):
        print(f" 파일 없음: {path}")
        return None
    img = cv2.imread(path)
    print(f" 이미지 로드 완료: {img.shape}")
    return img


#     2단계: 그레이스케일 변환     
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#     3단계: 대비 최대화     
def enhance_contrast(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    contrast = cv2.add(gray, tophat)
    contrast = cv2.subtract(contrast, blackhat)
    contrast = cv2.equalizeHist(contrast)
    return contrast


#     4단계: 적응형 임계처리     
def adaptive_threshold(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2)
    return thresh


#     5단계: 윤곽선 검출           
def detect_contours(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#     6단계: 시각화 및 저장         
def visualize_results(images, titles):
    plt.figure(figsize=(15, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


#     전체 파이프라인 함수         
def process_plate(filename, visualize=True):
    print(f"\n {filename} 처리 시작")

    original = load_plate_image(filename)
    if original is None:
        return

    gray = to_grayscale(original)
    enhanced = enhance_contrast(gray)
    thresh = adaptive_threshold(enhanced)
    contours = detect_contours(thresh)

    print(f" 윤곽선 수: {len(contours)}개")

    if visualize:
        contour_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_display, contours, -1, (0, 255, 0), 1)

        visualize_results(
            [gray, enhanced, thresh, contour_display],
            ['Grayscale', 'Enhanced', 'Threshold', 'Contours']
        )

    return thresh  # OCR을 위한 최종 흑백 이미지 반환

#     실행 예시                   
if __name__ == "__main__":
    process_plate("plate_02")