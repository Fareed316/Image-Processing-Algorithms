import cv2
import numpy as np
import math

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
			
img = cv2.imread("images/img1.jpg", 0 )

height = img.shape[0]
width = img.shape[1]
pixels = height*width

histo = histogram(img)
cum_hist = cumulative_histogram(histo)


for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		b = math.floor((cum_hist[a] *255) / pixels)
		img.itemset((i,j), b)

cv2.imwrite('outputs/histogram_eq.jpg', img)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


























