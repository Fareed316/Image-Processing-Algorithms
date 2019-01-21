import cv2
import numpy as np
from matplotlib import pyplot as plt

def convolve_np(X ,F):

	height = X.shape[0]
	width = X.shape[1]

	height_f = F.shape[0]
	width_f = F.shape[1]

	out = np.zeros((height, width))


	H = int((height_f - 1)/2)
	W = int((width_f -1 ) /2)

	for i in np.arange(H, height -H):
		for j in np.arange(W , width-W):
			sum = 0

			for k in np.arange(-H , H+1):
				for l in np.arange(-W, W+1):
					a = X[i+k, j+l]
					w = F[H+k, W+l]
					sum += (a*w)
			out[i,j] = sum
	return out


img = cv2.imread('images/cat.jpg',cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]


Hx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])

Hy = np.array([[-1, -1, -1],
               [0, 0, 0],
               [1, 1, 1]])



img_x = convolve_np(img , Hx) / 6.0
img_y = convolve_np(img , Hy) / 6.0

img_out = np.sqrt(np.power(img_x,2) + np.power(img_y, 2))


img_out = (img_out / np.max(img_out)) *255





cv2.imwrite('outputs/perwitt.jpg', img_out)

cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows() 
plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""
cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""





















