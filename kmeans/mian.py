import numpy as np
import cv2 as cv

img = cv.imread('G:\Programming\Python\kmeans\leaf.jpg')
img2 = img.reshape((-1,3))
img2 = np.float32(img2)

criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,10,1.0)
k = 4
attempts = 10
ret,label,center = cv.kmeans(img2,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imwrite('new.jpg',res2)