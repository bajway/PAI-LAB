import cv2
import numpy as np

image = np.zeros((300, 400, 3), dtype=np.uint8)

cv2.imshow('Blank Image', image)
cv2.waitKey(0)

cv2.rectangle(image, (50, 50), (200, 150), (0, 0, 255), -1)

cv2.imshow('Image with Rectangle', image)
cv2.waitKey(0)

cv2.circle(image, (300, 200), 50, (0, 255, 0), -1)

cv2.imshow('Image with Rectangle and Circle', image)
cv2.waitKey(0)

cv2.rectangle(image, (10, 10), (390, 290), (255, 0, 0), 3)

cv2.imshow('Image with All Shapes', image)
cv2.waitKey(0)

cv2.circle(image, (200, 150), 30, (255, 255, 0), 2)

cv2.imshow('Final Image with Shapes', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
