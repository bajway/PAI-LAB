import cv2

image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

cv2.imshow('Image 1', image1)
cv2.waitKey(0)

cv2.imshow('Image 2', image2)
cv2.waitKey(0)

image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

blended_image = cv2.add(image1, image2)

cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)

gray_blended = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Blended Image', gray_blended)
cv2.waitKey(0)

equalized_image = cv2.equalizeHist(gray_blended)

cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
