import cv2
import numpy as np


def histogram(img):
	height = img.shape[0]
	width = img.shape[1]

	hist = np.zeros((256))
# as its only till 255 , because grey scale images have value
#between 0-225
	for i in np.arange(height):
		for j in np.arange(width):
			a = img.item(i,j)
			hist[a] += 1

	print()
	return hist

def cumulative_histogram(hist):
	cum_hist = hist.copy()

	for i in np.arange(1,256):
		cum_hist[i] = cum_hist[i-1] + cum_hist[i]
	
	return cum_hist
			



img = cv2.imread("images/low_contrast_1.jpg", 0)

height = img.shape[0]
width = img.shape[1]

pixels = height * width

hist = histogram(img)
cum_hist = cumulative_histogram(hist)

p = 0.005

a_low = 0
for i in np.arange(256):
	if cum_hist[i] >= p*pixels:
		a_low = i
		break
a_high = 256

for i in np.arange(255,-1,-1):
	if cum_hist[i] <= (1-p)*pixels:
		a_high = i
		break

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		if a < a_low:
			b=0
		elif a>a_high:
			b=255
		else:
			b = (a - a_low)/(a_high- a_low) *255

		img.itemset((i,j),b)

cv2.imwrite("outputs/new_contrast_mod.jpg", img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(a_low)
print(a_high)






