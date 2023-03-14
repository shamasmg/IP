#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
image = cv2.imread('car3.jpg')


hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
s_ch = hsv_img[:, :, 1]  
thesh = cv2.threshold(s_ch, 5, 255, cv2.THRESH_BINARY)[1]  
thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1, upDiff=1) 

image[thesh == 128] = (0, 0, 255)  
#cv2.imwrite('tulips_red_bg.jpg', image) 
cv2.imshow(' ',image)
cv2.waitKey(0)


# In[4]:


import cv2
import numpy as np
image = cv2.imread('car3.jpg')


hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
s_ch = hsv_img[:, :, 1]  
thesh = cv2.threshold(s_ch, 5, 255, cv2.THRESH_BINARY)[1]  
thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1, upDiff=1) 

image[thesh == 128] = (0,255, 0)  
#cv2.imwrite('tulips_green_bg.jpg', image) 
cv2.imshow(' ',image)
cv2.waitKey(0)


# In[6]:


import cv2
import numpy as np
image = cv2.imread('car3.jpg')


hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
s_ch = hsv_img[:, :, 1]  
thesh = cv2.threshold(s_ch, 5, 255, cv2.THRESH_BINARY)[1]  
thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1, upDiff=1) 

image[thesh == 128] = (255,0, 0)  
#cv2.imwrite('tulips_blue_bg.jpg', image) 
cv2.imshow(' ',image)
cv2.waitKey(0)


# In[ ]:




