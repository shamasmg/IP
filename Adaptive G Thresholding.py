#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#BASSIC TRANFORMATION

import cv2 
import numpy as np 

image = cv2.imread('car3.jpg') 


height, width = image.shape[:2]
quarter_height, quarter_width = height/4, width/4

T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 
img_translation = cv2.warpAffine(image, T, (width, height)) 

cv2.imshow("Originalimage", image) 
cv2.imshow('Translation', img_translation)
cv2.imwrite('Translation.jpg', img_translation) 
cv2.waitKey() 

cv2.destroyAllWindows() 


# cv2.imshow("Nearest Neighbour", scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("Bilinear", scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("Bicubic", scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[ ]:


#ADAPTIVE-GAUSSIAN THRESHOLDING 

import cv2
import numpy as np

img = cv2.imread('cat.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray', img)

blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('blur', blur)

ret,th1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
cv2.imshow('Global', th1)
cv2.imwrite('Global.jpg',th1)


th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Adaptive Mean', th2)
cv2.imwrite('AM.jpg',th2)


th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Adaptive Gaussian', th3)
cv2.imwrite('AG.jpg',th3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:



#import libraries
import cv2
from matplotlib import pyplot as plt
  
# create figure
fig = plt.figure(figsize=(10, 10))
  
# setting values to rows and column variables
rows = 2
columns = 3
  
img = cv2.imread('flower2.png', cv2.IMREAD_GRAYSCALE)

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image


plt.imshow(img)
plt.axis('off')
plt.title("First")

fig.add_subplot(rows, columns, 2) 

blur = cv2.GaussianBlur(img,(5,5),0)

plt.imshow(blur)
plt.axis('off')
plt.title(" ")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 3)

ret,th1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
  
# showing image
plt.imshow(th1)
plt.axis('off')
plt.title(" ")

fig.add_subplot(rows, columns, 4)

th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)


plt.imshow(th2);
plt.axis('off');
plt.title(" ")
  
    
 # Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 5)

th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  
# showing image
plt.imshow(th3)
plt.axis('off')
plt.title(" ")


# In[ ]:



  
# This is not correct it is for knowlegde only
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('flower2.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=((10,10)))
plt.subplot(1,5,1);
plt.imshow(img);

blur = cv2.GaussianBlur(img,(5,5),0)

plt.figure(figsize=((10,10)))
plt.subplot(1,5,2);
plt.imshow(blur)

ret,th1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)

plt.figure(figsize=((10,10)))
plt.subplot(1,5,3);
plt.imshow(th1)


th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

plt.figure(figsize=((10,10)))
plt.subplot(1,5,4);
plt.imshow(th2)


th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

plt.figure(figsize=((10,10)))
plt.subplot(1,5,5);
plt.imshow(th3)


# In[ ]:


# FILTER

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('panda1.jpg',0)
# x = int(input())

# kernel = np.ones((x,x),np.float32)/pow(x,2)
# dst = cv2.blur(img,0,kernel,img,(-1,-1),False,cv2.BORDER_DEFAULT)
# #cv2.imwrite('dts.png', dst)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('color1.jpg')

blur = cv2.blur(img,(5,5))

plt.imshow(img),plt.title('Original')
plt.show()
cv2.imwrite('orig.png', img)

plt.xticks([]), plt.yticks([])
plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
cv2.imwrite('boxfil.png', blur)


# In[ ]:


#CONTOURS

import cv2 
import numpy as np 

image = cv2.imread('car1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
cv2.imshow('Canny Edges After Contouring', edged)
#cv2.imwrite('CannyFish.jpg', edged)  
cv2.waitKey(0) 

print("Number of Contours: " + str(len(contours))) 
  
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', image)
#cv2.imwrite('Contours.jpg', image) 
cv2.waitKey(0) 

image2 = cv2.imread('car1.jpg')
if len(contours) != 0:
	c = max(contours, key = cv2.contourArea)
cv2.drawContours(image2, c, -1, (0, 255, 0), 3)
cv2.imshow('Largest Contour', image2)
#cv2.imwrite('LargestContour.jpg', image2) 
cv2.waitKey(0) 

cv2.destroyAllWindows() 


# In[ ]:



import matplotlib.pyplot as plt
import cv2 
import numpy as np 

fig=plt.figure(figsize=(8,8))
row=2
column=2

fig.add_subplot(row,column,1)

image = cv2.imread('car2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
plt.imshow(image)



fig.add_subplot(row,column,2)

edged = cv2.Canny(gray, 30, 200) 

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
plt.imshow(edged)

fig.add_subplot(row,column,3)

print("Number of Contours: " + str(len(contours))) 
  
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
plt.imshow(image) 

fig.add_subplot(row,column,4)

image2 = cv2.imread('car1.jpg')
if len(contours) != 0:
	c = max(contours, key = cv2.contourArea)
cv2.drawContours(image2, c, -1, (0, 255, 0), 3)
plt.imshow(image2)


# In[ ]:


#DESCRETE FOURIER TRANSFORMATION

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('mask2.png',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#cv2.imwrite("idft.jpg", magnitude_spectrum)
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


c = 255/(np.log(1 + np.max(img))) 
log_transformed = c * np.log(1 + img) 
  
# Specify the data type. 
log_transformed = np.array(log_transformed, dtype = np.uint8) 

plt.imshow(log_transformed)
#cv2.imshow(log_transformed, "transformed")
#cv2.imwrite("idft1.jpg", log_transformed)
#cv2.waitkey(0)


# In[ ]:


#DESCRETE FOURIER TRANSFORMATION

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('apple1.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#cv2.imwrite("idft.jpg", magnitude_spectrum)
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


c = 255/(np.log(1 + np.max(img))) 
log_transformed = c * np.log(1 + img) 
  
# Specify the data type. 
log_transformed = np.array(log_transformed, dtype = np.uint8) 

plt.imshow(log_transformed)
#cv2.imshow(log_transformed, "transformed")
#cv2.imwrite("idft1.jpg", log_transformed)
##cv2.waitkey(0)


# In[ ]:


#ESTIMATING THE TRANSFORMATION

import cv2
import numpy as np

def piecewise(img,h,w):
	for i in range(h):
		for j in range(w):
			if(img[i][j] > 105 and img[i][j] < 165):
				img[i][j] =10


img = cv2.imread('apple1.jpg', 0)
(h,w) = img.shape[:2]

piecewise(img,h,w)

cv2.imshow("image",img)
#cv2.imwrite("image.jpg", img)
cv2.waitKey(0)


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def piecewise(img,h,w):
	for i in range(h):
		for j in range(w):
			if(img[i][j] > 105 and img[i][j] < 165):
				img[i][j] =10


img = cv2.imread('apple1.jpg', 0)

(h,w) = img.shape[:2]

piecewise(img,h,w)

plt.imshow(img)


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def piecewise(img,h,w):
	for i in range(h):
		for j in range(w):
			if(img[i][j] > 105 and img[i][j] < 165):
				img[i][j] =10

fig=plt.figure(figsize=(7,7))
row=1
columns=2

fig.add_subplot(row,columns,1)
img = cv2.imread('Est.jpg', 0)
plt.imshow(img)

(h,w) = img.shape[:2]

piecewise(img,h,w)

fig.add_subplot(row,columns,2)
plt.imshow(img)


# In[ ]:


#FITTING POLYGONS

import cv2 
import numpy as np

count=0
img = cv2.imread('polygons.jpg', cv2.IMREAD_GRAYSCALE)  
cv2.imshow('gray', img)
cv2.waitKey(0)

_,threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) 
cv2.imshow('Threshold Image', threshold)
cv2.imwrite('Threshold.jpg', threshold)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', img)
cv2.imwrite('Contours.jpg', img)
cv2.waitKey(0)

image2 = cv2.imread('polygons.jpg')
for contour in contours:
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour,0.05 * cv2.arcLength(contour, True), True)
	if len(approx) == 3	:
		cv2.drawContours(image2, contour, -1, (0, 0, 0), 3)
		count = count + 1

print("Number of Triangles in the image: " + str(count)) 
cv2.imshow('Triangles Detected', image2)
cv2.imwrite('Triangles.jpg', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()	


# In[ ]:


import cv2 
import numpy as np


fig=plt.figure(figsize=(7,7))
row=2
columns=2



count=0
img = cv2.imread('polygons.jpg', cv2.IMREAD_GRAYSCALE)  
#cv2.imshow('gray', img)
fig.add_subplot(row,columns,1)
plt.imshow(img)
plt.title("gray")



_,threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) 
#cv2.imshow('Threshold Image', threshold)
fig.add_subplot(row,columns,2)
plt.imshow(threshold)
plt.title("Threshold Image")

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
#cv2.imshow('Contours', img)
fig.add_subplot(row,columns,3)
plt.imshow(img)
plt.title("Contours")

image2 = cv2.imread('polygons.jpg')
for contour in contours:
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour,0.05 * cv2.arcLength(contour, True), True)
	if len(approx) == 3	:
		cv2.drawContours(image2, contour, -1, (0, 0, 0), 3)
		count = count + 1

print("Number of Triangles in the image: " + str(count)) 
#cv2.imshow('Triangles Detected', image2)
fig.add_subplot(row,columns,4)
plt.imshow(image2)
plt.title("Triangles Detected")


# In[ ]:


#Grabcut

import numpy as np
import cv2
from matplotlib import pyplot as plt

fig, ax = plt.subplots(3, 2, figsize=(10,10))

img = cv2.imread('bgi.jpg')
#cv2.imshow('img', img)
plt.subplot(1,2,1),plt.imshow(img)

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (10,100,200,300)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_cut = img*mask2[:,:,np.newaxis]

#cv2.imshow('img_cut', img_cut)
plt.subplot(1,2,2),plt.imshow(img_cut)
#cv2.imwrite('cut_messi.jpg', img_cut)
cv2.waitKey(0)


# In[5]:


#Shading correction
import cv2

img = cv2.imread('Est.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', grayImg)
cv2.waitKey(0)
filtersize = 513
gaussianImg = cv2.GaussianBlur(grayImg, (filtersize, filtersize), 128)
cv2.imshow('Converted Image', gaussianImg)
cv2.waitKey(0)
newImg = (grayImg-gaussianImg)
cv2.imshow('New Image', newImg)
#cv2.imwrite('Converted.png', newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




