import cv2

image = cv2.imread('image.png')

cv2.imshow('Original Image', image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)

new_size = (300, 200)
resized_image = cv2.resize(image, new_size)

cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)

resized_gray = cv2.resize(gray_image, new_size)

cv2.imshow('Resized Grayscale Image', resized_gray)
cv2.waitKey(0)

cv2.destroyAllWindows()
