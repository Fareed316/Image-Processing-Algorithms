import cv2
import numpy as np


img = cv2.imread('images/cables.jpg', 0)

height = img.shape[0]
width = img.shape[1]


threshold = 100

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)

		if a> threshold:
			b = 255
		else:
			b = 0
		img.itemset((i,j), b)

cv2.imwrite('outputs/cables_threshold.jpg', img)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
