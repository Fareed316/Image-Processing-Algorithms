import cv2
import math
import numpy as np


img1 = cv2.imread('images/cat.jpg', 0 )
img2 = cv2.imread('images/lion.jpg', 0 )

alpha = 0.25

height = img1.shape[0]
width = img1.shape[1]

for i in np.arange(height):
	for j in np.arange(width):
		a1 = img1.item(i,j)
		a2 = img2.item(i,j)

		b = alpha*a1 + (1-alpha)*a2
		img1.itemset((i,j), b)


cv2.imwrite('outputs/linear_mix.jpg', img1)

cv2.imshow('ah', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()