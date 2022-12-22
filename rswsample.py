#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
image=cv2.imread('flower1.png')
cv2.imshow('Display image',image)
cv2.imwrite('D:\img1.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import cv2 
image1=cv2.imread('flower2.png')
cv2.imshow('Display image',image1)
cv2.imwrite('D:\img2.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


new_image=cv2.rotate(image1,cv2.ROTATE_180)
cv2.imshow('Display image',new_image)
cv2.waitKey(0)


# In[4]:


import matplotlib.pyplot as plt
image=plt.imread('flower1.png')
plt.imshow(image)
plt.show()


# In[5]:


image.size


# In[6]:


h, w, c = image.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)


# In[7]:


image1.size


# In[8]:


print(image.shape)


# In[9]:


import cv2 
image1=cv2.imread('flower2.png')
cv2.imshow('Display image',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


print(image1.shape)


# In[11]:


h, w, c = image1.shape
print('width:  ', w)
print('height: ', h)
print('channel:', c)


# In[37]:


ret, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 
# converting to its binary form
bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
 
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[47]:


ret, bw_img = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
 
# converting to its binary form
bw = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
 
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[49]:


down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.destroyAllWindows()


# In[40]:


up_width = 600
up_height = 400
up_points = (up_width, up_height)
# resize the image
resized_up = cv2.resize(image, up_points, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey()
cv2.destroyAllWindows()


# In[41]:


print('The Shape of the image is:',image.shape)
print('The image as array is:')
print(image)


# In[36]:


from PIL import Image
import numpy as np
w, h = 512, 512
data = np. zeros((h, w, 3),dtype=np.uint8)
data[120:256, 120:256] = [255, 0, 0] # red patch in upper left.
img = Image.fromarray(data,'RGB')
img.save('my.png')

images=plt.imread('my.png')
plt.imshow(images)
plt.show()


# In[17]:


import matplotlib.pyplot as plt
image=plt.imread('flower1.png')
plt.imshow(image)
plt.show()


# In[18]:


import matplotlib.pyplot as plt
image1=plt.imread('flower2.png')
plt.imshow(image1)
plt.show()


# In[ ]:





# In[19]:


import matplotlib.pyplot as plt
import cv2
image=cv2.imread('flower2.png',0)
resized = cv2.resize(image,(500,500))
 #display the two resized images
_,ax = plt.subplots(1,2)
ax[0].imshow(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
ax[1].axis('off')


# In[20]:


import matplotlib.pyplot as plt
import cv2
image=cv2.imread('color1.jpg')
resized = cv2.resize(image,(500,500))
 #display the two resized images
color=cv2.cvtColor(resized,cv2.COLOR_RGB2BGR)
cv2.imshow('color1.jpg',color)
cv2.waitKey()
cv2.destroyAllWindows()


# In[42]:


import matplotlib.pyplot as plt
import cv2
image2=cv2.imread('color1.jpg')
resized = cv2.resize(image2,(500,500))
 #display the two resized images
_,ax = plt.subplots(1,2)
ax[0].imshow(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2HLS))
ax[1].axis('off')


# In[32]:


import cv2
image2=cv2.imread('color1.jpg')
color=cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
cv2.imshow('display BGR',color)
cv2.waitKey()
cv2.destroyAllWindows()


# In[33]:


import cv2
image3=cv2.imread('color1.jpg')
color1=cv2.cvtColor(image3,cv2.COLOR_BGR2HLS)
cv2.imshow('display HLS',color1)
cv2.waitKey()
cv2.destroyAllWindows()


# In[34]:


color1=cv2.cvtColor(image3,cv2.COLOR_BGR2LAB)
cv2.imshow('display LAB',color1)
cv2.waitKey()
cv2.destroyAllWindows()


# In[35]:


color1=cv2.cvtColor(image3,cv2.COLOR_BGR2YUV)
cv2.imshow('display YUV',color1)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




