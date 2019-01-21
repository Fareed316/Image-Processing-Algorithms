import cv2
import numpy as np


img = cv2.imread("images/low_contrast_1.jpg", 0)

height = img.shape[0]
width = img.shape[1]

min = 255
max = 0

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		if a > max:
			max = a
		if a < min:
			min = a

print(max)
print(min)
for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		b = float(a-min)/ (max-min) *255
		img.itemset((i,j), b)

cv2.imwrite('outputs/low_contrast_1_auto_threshold.jpg', img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

