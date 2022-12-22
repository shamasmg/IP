#!/usr/bin/env python
# coding: utf-8

# In[3]:



import cv2
img1 = cv2.imread('car1.jpg')
img2 = cv2.imread('car2.jpg')
and_img = cv2.bitwise_and(img1,img2)
cv2.imshow('Bitwise AND Image', and_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read an input image as a gray image
img = cv2.imread('bfly3.png',0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:400, 150:600] = 255

# compute the bitwise AND using the mask
masked_img = cv2.bitwise_and(img,img,mask = mask)

# display the input image, mask, and the output image
plt.subplot(221), plt.imshow(img, 'gray'), plt.title("Original Image")
plt.subplot(222), plt.imshow(mask,'gray'), plt.title("Mask")
plt.subplot(223), plt.imshow(masked_img, 'gray'), plt.title("Output Image")
plt.show()


# In[10]:


import cv2

# read two images. The size of both images must be the same.
img1 = cv2.imread('bird1.png')
img2 = cv2.imread('bird2.png')

# compute bitwise OR on both images
or_img = cv2.bitwise_or(img1,img2)

# display the computed bitwise OR image
cv2.imshow('Bitwise OR Image', or_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:


import cv2
img1 = cv2.imread('car1.jpg')
img2 = cv2.imread('car2.jpg')
or_img = cv2.bitwise_or(img1,img2)
cv2.imshow('Bitwise AND Image', or_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[12]:


import cv2

# read two input images.
# The size of both images must be the same.
img1 = cv2.imread('bird1.png')
img2 = cv2.imread('bird2.png')

# compute bitwise AND on both images
and_img = cv2.bitwise_and(img1,img2)

# display the computed bitwise AND image
cv2.imshow('Bitwise AND Image', and_img)
or_img = cv2.bitwise_or(img1,img2)

# display the computed bitwise OR image
cv2.imshow('Bitwise OR Image', or_img)
xor_img = cv2.bitwise_xor(img1,img2)

# display the computed bitwise XOR image
cv2.imshow('Bitwise XOR Image', xor_img)
bitwise_not = cv2.bitwise_not(img1,img2)

cv2.imshow("bitwise_not", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


import cv2 
img=cv2.imread("bfly3.png")
b,g,r=cv2.split(img)
cv2.imshow("Red",r)
cv2.imshow("Green",g)
cv2.imshow("Blue",b)
cv2.waitKey(0)


# In[15]:


import cv2 
img=cv2.imread("flower2.png")
b,g,r=cv2.split(img)
cv2.imshow("Red",r)
cv2.imshow("Green",g)
cv2.imshow("Blue",b)
cv2.waitKey(0)


# In[18]:




import cv2 
import numpy
 
img = cv2.imread('color1.jpg') 
 
blue, green, red = cv2.split(img)
 
 
zeros = numpy.zeros(blue.shape, numpy.uint8)
 
blueBGR = cv2.merge((blue,zeros,zeros))
greenBGR = cv2.merge((zeros,green,zeros))
redBGR = cv2.merge((zeros,zeros,red))
 
 
cv2.imshow('blue BGR', blueBGR)
cv2.imshow('green BGR', greenBGR)
cv2.imshow('red BGR', redBGR)
 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


from PIL import Image,ImageStat
im=Image.open('bird1.png')
stat=ImageStat.Stat(im)
print(stat.median)


# In[25]:


from PIL import Image,ImageStat
im=Image.open('bird1.png')
stat=ImageStat.Stat(im)
print(stat.stddev)


# In[26]:


from PIL import Image,ImageStat
im=Image.open('bird1.png')
stat=ImageStat.Stat(im)
print(stat.mean)


# In[38]:


#RGB Channels
import matplotlib.pyplot as plt
im1=Image.open("car1.jpg")
ch_r,ch_g,ch_b=im1.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1);
plt.imshow(ch_r,cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2);
plt.imshow(ch_g,cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3);
plt.imshow(ch_b,cmap=plt.cm.Blues);plt.axis('off')
plt.tight_layout()
plt.show()


# In[40]:


#RGB Channels
import matplotlib.pyplot as plt
im1=Image.open("color1.jpg")
ch_r,ch_g,ch_b=im1.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1);
plt.imshow(ch_r,cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2);
plt.imshow(ch_g,cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3);
plt.imshow(ch_b,cmap=plt.cm.Blues);plt.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




