import cv2
import numpy as np

img = cv2.imread('images/plane_noisy.png', 0)
img_copy = img.copy()

height = img.shape[0]
width = img.shape[1]

for i in np.arange(3, height-3):
	for j in np.arange(3, width-3):
		neighbours = []
		for k in np.arange(-3,4):
			for l in np.arange(-3,4):
				a = img.item(i+k, j+l)
				neighbours.append(a)
		neighbours.sort()
		median = neighbours[24]
		img_copy.itemset((i,j), median)

cv2.imwrite('outputs/median.jpg', img_copy)

cv2.imshow('image', img_copy)

cv2.waitKey(0)

cv2.destroyAllWindows()