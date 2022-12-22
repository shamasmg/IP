#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np

# load image and get dimensions
img = cv2.imread("imgbg.png")
h, w, c = img.shape

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# do floodfill
result = img.copy()
cv2.floodFill(result, mask, (0,0), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (38,313), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (363,345), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (619,342), (255,255,255), (3,151,65), (3,151,65), flags=8)

# write result to disk
cv2.imwrite("me.png", result)

# display it
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


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


# In[2]:


import turtle
colors=['red','green','yellow','pink','black']
t=turtle.Pen()
turtle.bgcolor('gray')
for x in range(300):
    t.pencolor(colors[x%5])
    t.width(x//100 + 1)
    t.forward(x)
    t.left(59)


# In[ ]:




