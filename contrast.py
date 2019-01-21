import cv2
import numpy as np
import math 

img = cv2.imread("images/low_contrast_1.jpg", 0)

height = img.shape[0]
width = img.shape[1]

contrast = 1.9

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		b = math.ceil(a*contrast)
		if b >225 :
			b=225
		img.itemset((i,j),b)

cv2.imwrite("outputs/high_contrast.jpg", img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()