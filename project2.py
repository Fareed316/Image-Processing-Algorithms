import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import os


sizeOfInput = 0
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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




  

def prewitt(gray):
	height = gray.shape[0]
	width = gray.shape[1]


	Hx = np.array([[-1, 0, 1],
	               [-1, 0, 1],
	               [-1, 0, 1]])

	Hy = np.array([[-1, -1, -1],
	               [0, 0, 0],
	               [1, 1, 1]])


	img_x = convolve_np(gray , Hx) / 6.0
	img_y = convolve_np(gray , Hy) / 6.0

	gradientMagnitude = np.sqrt(np.power(img_x,2) + np.power(img_y, 2))


	gradientMagnitude = (gradientMagnitude / np.max(gradientMagnitude)) *255

	gradientAngle = np.zeros((height, width))
	for i in range(1, height - 1, 1):
		for j in range(1, width - 1, 1):
			if img_x[i][j] == 0 and img_y[i][j] == 0:
				gradientAngle[i][j] = 0
			elif img_x[i][j] == 0 and img_y[i][j] != 0:
				gradientAngle[i][j] = 90
			else:
				x = math.degrees(math.atan(img_y[i][j] / img_x[i][j]))
				if x < 0:
					x = 360 + x
				if x >= 170 or x < 350:
					x = x - 180
				gradientAngle[i][j] = x


	return gradientAngle ,gradientMagnitude

def hogCell(gradientAngle, gradientMagnitude):
	imgHeight = gradientAngle.shape[0]
	imgWidth = gradientAngle.shape[1]

	cellHistogram = np.zeros((int(imgHeight / 8), int(imgWidth * 9 / 8)))

	tempHist = np.zeros((1, 9))

	for i in range(0, imgHeight - 7, 8):
		for j in range(0, imgWidth - 7, 8):
			tempHist = tempHist * 0
			for k in range(8):
				for l in range(8):
					angle = gradientAngle[i + k][j + l]
					if -10 <= angle < 0:
						dist = 0 - angle
						tempHist[0][0] = tempHist[0][0] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][8] = tempHist[0][8] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 0 <= angle < 20:
						dist = angle
						tempHist[0][0] = tempHist[0][0] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][1] = tempHist[0][1] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 20 <= angle < 40:
						dist = angle - 20
						tempHist[0][1] = tempHist[0][1] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][2] = tempHist[0][2] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 40 <= angle < 60:
						dist = angle - 40
						tempHist[0][2] = tempHist[0][2] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][3] = tempHist[0][3] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 60 <= angle < 80:
						dist = angle - 60
						tempHist[0][3] = tempHist[0][3] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][4] = tempHist[0][4] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 80 <= angle < 100:
						dist = angle - 80
						tempHist[0][4] = tempHist[0][4] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][5] = tempHist[0][5] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 100 <= angle < 120:
						dist = angle - 100
						tempHist[0][6] = tempHist[0][6] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 120 <= angle < 140:
						dist = angle - 120
						tempHist[0][6] = tempHist[0][6] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][7] = tempHist[0][7] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 140 <= angle < 160:
						dist = angle - 140
						tempHist[0][7] = tempHist[0][7] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][8] = tempHist[0][8] + dist * gradientMagnitude[i + k][j + l] / 20
					elif 160 <= angle < 170:
						dist = angle - 160
						tempHist[0][8] = tempHist[0][8] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
						tempHist[0][0] = tempHist[0][0] + dist * gradientMagnitude[i + k][j + l] / 20
						cellHistogram[int(i / 8)][int(j * 9 / 8):int(j * 9 / 8 + 9)] = tempHist
	return cellHistogram

def hogBlock(cellHistogram):
	imgHeight = cellHistogram.shape[0]
	imgWidth = cellHistogram.shape[1]

	blockHistogram = np.empty((int(imgHeight - 1), int((imgWidth / 9 - 1) * 36)))
	tempHistogram = np.zeros((1, 36))

	for i in range(0, imgHeight - 1, 1):
		for j in range(0, imgWidth - 17, 9):
			l2Norm = 0
			for k in range(2):
				for l in range(18):
					l2Norm = l2Norm + math.pow(cellHistogram[i + k][j + l], 2)
			l2Norm = math.sqrt(l2Norm)
			x = 0
			for k in range(2):
				for l in range(18):
					if l2Norm == 0:
						tempHistogram[0][x] = 0
					else:
						tempHistogram[0][x] = cellHistogram[i + k][j + l] / l2Norm
					x = x + 1
			blockHistogram[i][int(j * 36 / 9):int(j * 36 / 9 + 36)] = tempHistogram
	blockHistogram = blockHistogram#.flatten()
	sizeOfInput = blockHistogram.shape[0]
	return blockHistogram

def reLu(num):
    if num <= 0:
        return 0
    else:
        return num


def reLuDeriv(num):
    if num <= 0:
        return 0
    else:
        return 1


img = cv2.imread('Test_Positive/crop_000010b.bmp')     
gray = rgb2gray(img)  

grad_angle , grad_mag= prewitt(gray)

cell_histo = hogCell(grad_mag,grad_angle)
tempHist = hogBlock(cell_histo)
print(tempHist)

imS = cv2.resize(tempHist, (96, 160))

cv2.imshow('image', imS)

cv2.waitKey(0)
plt.figure(figsize=(1,1))
plt.imshow(tempHist, cmap = 'gray', interpolation = 'bicubic')

plt.xticks([]), plt.yticks([]) 
cv2.destroyAllWindows()
cv2.imwrite('outputs/FUCCCKKK.jpg', tempHist)

#cv2.imshow('image', img_out)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
#plt.imshow(img_out, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
"""
cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""





















