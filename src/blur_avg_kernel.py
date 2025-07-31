import cv2
import numpy as np

img = cv2.imread('../img/paper.jpg')

kernel = np.ones((5,5))/5**2

blured = cv2.filter2D(img, -1, kernel)

cv2.imshow('origin', img)
cv2.imshow('avrg blur', blured)
cv2.waitKey(0)
cv2.destroyAllWindows()
