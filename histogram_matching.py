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

img = cv2.imread('images/img1.jpg', 0)
img_ref = cv2.imread('images/cat.jpg', 0)

height  = img.shape[0]
width = img.shape[1]
pixels = height * width 

height_ref = img_ref.shape[0]
width_ref = img_ref.shape[1]
pixels_ref = height_ref * width_ref


histo = histogram(img)
cum_hist = cumulative_histogram(histo)

histo_ref = histogram(img_ref)
cum_hist_ref = cumulative_histogram(histo_ref)

prob_cum_hist = cum_hist / pixels
prob_cum_hist_ref = cum_hist_ref / pixels_ref

k = 256

new_values = np.zeros(k)

for a in np.arange(k):
	j = k-1
	while True:
		new_values[a] = j
		j = j-1
		if j<0 or prob_cum_hist[a] > prob_cum_hist_ref [j]:
			break

for i in np.arange(height):
	for j in np.arange(width):
		a = img.item(i,j)
		b = new_values[a]
		img.itemset((i,j), b)

cv2.imwrite('outputs/histo_match.jpg', img)

cv2.imshow('fk me baby', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






















