import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class HOG:
    def imagesNames(self):
        images = []
        positiveImg = os.listdir('./Train_Positive')
        negativeImg = os.listdir('./Train_Negative')
        for i in range(10):
            images.append('/Train_Positive/' + positiveImg[i])
            images.append('/Train_Negative/' + negativeImg[i])

        

        return images

    def caller(self, images):
        obj1 = ImageProcessing()       
        testImages = []
        positiveTest = os.listdir('./Test_Positive')
        negativeTest = os.listdir('./Test_Neg')
        for i in range(5):
            testImages.append('/Test_Positive/' + positiveTest[i])
            testImages.append('/Test_Neg/' + negativeTest[i])
        for i in range(10):
            imgTest = cv2.imread('.' + testImages[i])
            imageTest = obj1.imageGrayscale(imgTest)
            gradientMagnitudeTest, gradientAngleTest = obj1.prewitt(imageTest)
            cellHistogramTest = obj1.hogCell(gradientMagnitudeTest, gradientAngleTest)
            blockHistogramTest = obj1.hogBlock(cellHistogramTest)
            print(blockHistogramTest)
            plt.imshow(blockHistogramTest, cmap = 'gray', interpolation = 'bicubic')
            cv2.imshow('image', blockHistogramTest)




class ImageProcessing:

    sizeOfInput = 0

    def imageGrayscale(self, img):
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        image = np.zeros((imgHeight, imgWidth))
        for i in range(imgHeight):
            for j in range(imgWidth):
                image[i][j] = round(img[i][j][2] * 0.299 + img[i][j][1] * 0.587 + img[i][j][0] * 0.114)
        return image

    def prewitt(self, img):
        prewittHorizontal = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])

        prewittVertical = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        horizontalGradient = np.zeros((imgHeight, imgWidth))
        verticalGradient = np.zeros((imgHeight, imgWidth))

        for i in range(1, imgHeight - 1, 1):
            for j in range(1, imgWidth - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + ((img[i - 1 + k][j - 1 + l]) * prewittHorizontal[k][l])
                horizontalGradient[i][j] = x / 3

        for i in range(1, imgHeight - 1, 1):
            for j in range(1, imgWidth - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + (img[i - 1 + k][j - 1 + l] * prewittVertical[k][l])
                verticalGradient[i][j] = x / 3

        gradientAngle = np.zeros((imgHeight, imgWidth))

        for i in range(1, imgHeight - 1, 1):
            for j in range(1, imgWidth - 1, 1):
                if horizontalGradient[i][j] == 0 and verticalGradient[i][j] == 0:
                    gradientAngle[i][j] = 0
                elif horizontalGradient[i][j] == 0 and verticalGradient[i][j] != 0:
                    gradientAngle[i][j] = 90
                else:
                    x = math.degrees(math.atan(verticalGradient[i][j] / horizontalGradient[i][j]))
                    if x < 0:
                        x = 360 + x
                    if x >= 170 or x < 350:
                        x = x - 180
                    gradientAngle[i][j] = x

        gradientMagnitude = np.zeros((imgHeight, imgWidth), dtype='int')

        for i in range(1, imgHeight - 1, 1):
            for j in range(1, imgWidth - 1, 1):
                x = math.pow(horizontalGradient[i][j], 2) + math.pow(verticalGradient[i][j], 2)
                gradientMagnitude[i][j] = int(round(math.sqrt(x / 2)))
        return gradientAngle, gradientMagnitude

    def hogCell(self, gradientAngle, gradientMagnitude):
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
                            tempHist[0][5] = tempHist[0][5] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
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

    def hogBlock(self, cellHistogram):
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
        ImageProcessing.sizeOfInput = blockHistogram.shape[0]
        return blockHistogram




def main():
    obj = HOG()
    a= obj.imagesNames()
    obj.caller(a)



if __name__ == "__main__":
    main()