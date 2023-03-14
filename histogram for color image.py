#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
img=Image.open('car3.jpg')
pl=img.histogram()
plt.bar(range(256),pl[:256],color='r',alpha=0.5)
plt.bar(range(256),pl[256:2*256],color='g',alpha=0.4)
plt.bar(range(256),pl[2*256:],color='b',alpha=0.3)
plt.show()


# In[3]:


#histogram for grayscale image

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lowCon.png',0)
plt.hist(img.ravel(),256,[0,256]) 

plt.show() 
#plt.savefig('hist.png')

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))

cv2.imshow('Equalized Image',res)
#cv2.imwrite('Equalized Image.png',res)

plt.hist(res.ravel(),256,[0,256]) 

plt.show() 
#plt.savefig('equal-hist.png')

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


from PIL import Image
import matplotlib.pyplot as plt
img=Image.open('color1.jpg')
pl=img.histogram()
plt.bar(range(256),pl[:256],color='r',alpha=0.5)
plt.bar(range(256),pl[256:2*256],color='g',alpha=0.4)
plt.bar(range(256),pl[2*256:],color='b',alpha=0.3)
plt.show()


# In[ ]:




