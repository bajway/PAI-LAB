import cv2
import numpy as np

image = cv2.imread('image.png')

target_width = 800
target_height = 600
resized_image = cv2.resize(image, (target_width, target_height))

text = "This is Computer Vision Lab"
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(resized_image, text, (50, 150), font, 1.5, (0, 0, 255), 2)

cv2.imshow('Text on Loaded Image', resized_image)
cv2.waitKey(0)

blank_image = np.zeros((300, 800, 3), dtype=np.uint8)

cv2.putText(blank_image, "Hello OpenCV!", (50, 100), font, 2, (0, 255, 0), 3)

cv2.putText(blank_image, "Image Processing", (100, 200), font, 1.5, (255, 255, 255), 2)

cv2.imshow('Text on Blank Image', blank_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
