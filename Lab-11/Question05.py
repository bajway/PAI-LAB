import cv2

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('Original Grayscale Image', image)
cv2.waitKey(0)

ret, thresholded_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

cv2.imshow('Threshold Image', thresholded_image)
cv2.waitKey(0)

center = (thresholded_image.shape[1] / 2, thresholded_image.shape[0] / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 60, 1)

rotated_image = cv2.warpAffine(thresholded_image, rotation_matrix, (thresholded_image.shape[1], thresholded_image.shape[0]))

cv2.imshow('Rotated Thresholded Image', rotated_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
