#!/usr/bin/env python
# coding: utf-8

# In[1]:


#thresholding

import cv2

from matplotlib import pyplot as plt
img = cv2.imread('dog.jpg',0)

# Apply global (simple) thresholding on image
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Apply Otsu's thresholding on image
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

titles = ['Original Image','Global Thresholding (v=127)',"Otsu's Thresholding",'Gaussian Filter + Otsu']
images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# In[2]:


#thresholding
import cv2

from matplotlib import pyplot as plt
img = cv2.imread('fish.jpg',0)

# Apply global (simple) thresholding on image
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Apply Otsu's thresholding on image
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

titles = ['Original Image','Global Thresholding (v=127)',"Otsu's Thresholding",'Gaussian Filter + Otsu']
images = [img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# In[ ]:


#Otsus Thresholding

import cv2
import numpy as np

img = cv2.imread('noisy2.png', cv2.IMREAD_GRAYSCALE)  
cv2.imshow('gray', img)
#cv2.imwrite('gray1.jpg',img)

blur = cv2.GaussianBlur(img,(7,7),0)
cv2.imshow('blur', blur)
#cv2.imwrite('blur1.jpg',img)

x,threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary threshold', threshold)
#cv2.imwrite('binarythresh1.jpg',img)

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsus Thresholding', th2)
#cv2.imwrite('Otsus.jpg',img)

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsus Thresholding gaussian', th3)
#cv2.imwrite('Otsus.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


#Otsus Thresholding
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('dog.jpg',0)
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.figure(figsize=(10,10))
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:


#Otsus Thresholding
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('noisy2.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.figure(figsize=(12,12))
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:


#Otsus Thresholding

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('noisy3.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.figure(figsize=(12,12))
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()


# In[4]:


#ADAPTIVE-GAUSSIAN THRESHOLDING 

import cv2
import numpy as np

img = cv2.imread('noisy2.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray', img)

blur = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('blur', blur)

ret,th1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
cv2.imshow('Global', th1)
#cv2.imwrite('Global.jpg',th1)


th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Adaptive Mean', th2)
#cv2.imwrite('AM.jpg',th2)


th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('Adaptive Gaussian', th3)
#cv2.imwrite('AG.jpg',th3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


#global thrsholding manual

import cv2
import numpy as np
# Load an image in the greyscale
img = cv2.imread('fish.jpg',cv2.IMREAD_GRAYSCALE)

def global_threshold(image, thres_value, val_high, val_low):
    img = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > thres_value:
                img[i,j] = val_high
            else:
                img[i,j] = val_low
    return img
def thres_finder(img, thres=20,delta_T=1.0):
    
# Step-2: Divide the images in two parts
    x_low, y_low = np.where(img<=thres)
    x_high, y_high = np.where(img>thres)
   
# Step-3: Find the mean of two parts

    mean_low = np.mean(img[x_low,y_low])
    mean_high = np.mean(img[x_high,y_high])
   
# Step-4: Calculate the new threshold

    new_thres = (mean_low + mean_high)/2
   
# Step-5: Stopping criteria, otherwise iterate

    if abs(new_thres-thres)< delta_T:
         return new_thres
    else:
        return thres_finder(img, thres=new_thres,delta_T=1.0)

# apply threshold finder
vv1 = thres_finder(img, thres=30,delta_T=1.0)
# threshold the image
   
ret, thresh = cv2.threshold(img,vv1,255,cv2.THRESH_BINARY)
out = cv2.hconcat([img,thresh])
cv2.imshow('threshold',out)
cv2.waitKey(0)


# In[3]:


#ADAPTIVE THRESHOLDING MANUAL
(#adaptive)
# Description: This is an algorithm built to perform adaptive thresholding of an
# image. This algorithm basically works as follows,
# 1. Take the input image and resize for faster outputs.
# 2. Integrate the image such that each pixel represents the sum computed both  
#    above and behind the current pixel.
   
# 3. For the second iteration on the image, (Sample_window_1)x(Sample_window_2)
#    average is calculated for each pixel by utilizing the computed integral
#    image.
   
# 4. Based on the threshold value if the intensity is lesser the pixel value is
#    set to 0 and 255 if higher.
   
# """

import numpy as np
import cv2

number = 1
class adaptive_threshold:
   
    def __init__(self,image):
        image =cv2.imread("orig.png")
        self.input_image = cv2.resize(image,(256,196))
        self.height,self.width,_ = self.input_image.shape    
        self.sample_window_1 = self.width/12
        self.sample_window_2 = self.sample_window_1/2
        self.threshold = 68
        self.integrated_image = np.zeros_like(self.input_image, dtype=np.uint32)
        self.output_image = np.zeros_like(self.input_image)
        self.main()
   
    def integrate_image(self):
        for column in range(self.width):
            for row in range(self.height):
                self.integrated_image[row,column] = self.input_image[0:row,0:column].sum()
   
    def the_algorithm(self):
        for column in range(self.width):
            for row in range(self.height):
                y1 = round(max(row-self.sample_window_2,0))
                y2 = round(min(row+self.sample_window_2, self.height-1))
                x1 = round(max(column-self.sample_window_2,0))
                x2 = round(min(column+self.sample_window_2,self.width-1))
               
                count = (y2-y1)*(x2-x1)
               
                total = self.integrated_image[y2,x2]-self.integrated_image[y1,x2]-self.integrated_image[y2,x1]+self.integrated_image[y1,x1]          
               
                if np.all(self.input_image[row,column]*count < total*(100-self.threshold)/100):
                    self.output_image[row,column] = 0
                else:
                    self.output_image[row,column] = 255
           
    def main(self):
        self.integrate_image()
        self.the_algorithm()
        self.display_save()        
       
    def display_save(self):
        global number
        cv2.imshow('Image',self.input_image)
        cv2.imshow('Output',self.output_image)
        filename = 'output-'+str(number)+'.jpg'
        cv2.imwrite(filename,self.output_image)
        number+=1
        key = cv2.waitKey(0)

image1 = cv2.imread('testImage0.pgm')
image2 = cv2.imread('testImage1.pgm')
filter_ = adaptive_threshold(image1)
filter_ = adaptive_threshold(image2)


# In[2]:


import numpy as np
import cv2
from PIL import Image

def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1
im =np.array(Image.open('cat.png').convert('L')
#im = np.random.randint(0,255, size = (50,50))

     # testing all thresholds from 0 to the maximum of the image
             threshold_range = range(np.max(im)+1)
             criterias = [compute_otsu_criteria(im, th) for th in threshold_range]

    # best threshold is the one minimizing the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]
    cv2.imshow('OTSU',best_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




