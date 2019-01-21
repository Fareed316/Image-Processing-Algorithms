import cv2 
import numpy as np 


img = cv2.imread('images/cat.jpg', 0 )
img_copy1 = img.copy()
img_copy2 = img.copy()

height = img.shape[0]
width = img.shape[1]



for i in np.arange(3,height-3):
	for j in np.arange(3, width-3):
		max = 0
		min = 255
		for k in np.arange(-3,4):
			for l in np.arange(-3,4):
				a = img.item(i+k, j+l)
				if a>max:
					max = a
				if a<min:
					min = a
		b = max
		d = min
		img_copy1.itemset((i,j), b)
		img_copy2.itemset((i,j), d)
		
cv2.imwrite('outputs/max_2.jpg', img_copy1)
cv2.imwrite('outputs/min_2.jpg', img_copy2)

cv2.imshow('image_max', img_copy1)
cv2.imshow('image_min', img_copy2)

cv2.waitKey(0)
cv2.closeAllWindows()