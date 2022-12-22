#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
# Load the image
img = cv2.imread('car2.jpg')
# Check the datatype of the image
print(img.dtype)
# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img
# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)


# In[2]:


import cv2
from matplotlib import pyplot as plt
img = cv2.imread('car3.jpg',0)
 
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[3]:


# importing required libraries of opencv
import cv2

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread('car2.jpg',0)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


# In[4]:


import cv2
from matplotlib import pyplot as plt
img = cv2.imread('car2.jpg',0)
 
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('car3.jpg', -1)
cv2.imshow('car',img)

color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break              
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




