import cv2
import numpy as np

image = cv2.imread('image.jpg')

cv2.imshow('Original Image', image)
cv2.waitKey(0)

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow('Gaussian Blurred Image', blurred_image)
cv2.waitKey(0)

roi = image[100:300, 150:350]

cv2.imshow('Cropped Image', roi)
cv2.waitKey(0)

blurred_roi = cv2.GaussianBlur(roi, (7, 7), 0)

cv2.imshow('Blurred Cropped Image', blurred_roi)
cv2.waitKey(0)

cv2.destroyAllWindows()
