import cv2
import math
import numpy as np

img = cv2.imread('images/cat.jpg', 0)
img_copy = img.copy()

height = img.shape[0]
width = img.shape[1]

gauss = (1.0/57) * np.array(
        [[0, 1, 2, 1, 0],
        [1, 3, 5, 3, 1],
        [2, 5, 9, 5, 2],
        [1, 3, 5, 3, 1],
        [0, 1, 2, 1, 0]]) 

for i in np.arange(2, height-2):
	for j in np.arange(2,width-2):
		sum = 0 
		for k in np.arange(-2,3):
			for l in np.arange(-2,3):
				a =img.item(i+k,j+l)
				b = gauss[2+k,2+l]
				sum = sum + (a*b)
		img_copy.itemset((i,j), sum)

cv2.imwrite('outputs/gauss.jpg', img_copy)

cv2.imshow('Image', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()