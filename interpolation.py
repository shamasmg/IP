#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2
import numpy as np

img = cv2.imread('bird2.png')



near_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_NEAREST)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


import cv2
import numpy as np

img = cv2.imread('bird2.png')
bilinear_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_LINEAR)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


import cv2
import numpy as np

img = cv2.imread('bird2.png')
bicubic_img = cv2.resize(img,None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




