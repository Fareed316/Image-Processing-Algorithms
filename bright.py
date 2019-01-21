import cv2
import numpy as np
import math

img = cv2.imread("images/jet.jpg", 0)

height = img.shape[0]
width = img.shape[1]

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		b = a+50
		if b >255 :
			b= 255
		img.itemset((i,j), b)


cv2.imwrite("outputs/more_bright_jet.jpg", img)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
