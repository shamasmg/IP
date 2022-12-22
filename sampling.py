#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
img1 = cv2.imread('bird2.png', 0)
[m, n] = img1.shape
print('Image Shape:', m, n)
    
f = 4
img2 = np.zeros((m//f, n//f), dtype=np.int)
for i in range(0, m, f):
    for j in range(0, n, f):
        try:
 
            img2[i//f][j//f] = img1[i][j]
        except IndexError:
            pass
        
print('Down Sampled Image:')
plt.imshow(img2)
 


# In[2]:


# Up sampling
 
# Create matrix of zeros to store the upsampled image
img3 = np.zeros((m, n), dtype=np.int)
# new size
for i in range(0, m-1, f):
    for j in range(0, n-1, f):
        img3[i, j] = img2[i//f][j//f]
 
# Nearest neighbour interpolation-Replication
# Replicating rows
 
for i in range(1, m-(f-1), f):
    for j in range(0, n-(f-1)):
        img3[i:i+(f-1), j] = img3[i-1, j]
 
# Replicating columns
for i in range(0, m-1):
    for j in range(1, n-1, f):
        img3[i, j:j+(f-1)] = img3[i, j-1]
 
# Plot the up sampled image
print('Up Sampled Image:')
plt.imshow(img3)


# In[3]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
img1 = cv2.imread('flower2.png', 0)
[m, n] = img1.shape
print('Image Shape:', m, n)
    
f = 4
img2 = np.zeros((m//f, n//f), dtype=np.int)
for i in range(0, m, f):
    for j in range(0, n, f):
        try:
 
            img2[i//f][j//f] = img1[i][j]
        except IndexError:
            pass
        
print('Down Sampled Image:')
plt.imshow(img2, cmap="gray")
 


# In[4]:


# Up sampling
 
# Create matrix of zeros to store the upsampled image
img3 = np.zeros((m, n), dtype=np.int)
# new size
for i in range(0, m-1, f):
    for j in range(0, n-1, f):
        img3[i, j] = img2[i//f][j//f]
 
# Nearest neighbour interpolation-Replication
# Replicating rows
 
for i in range(1, m-(f-1), f):
    for j in range(0, n-(f-1)):
        img3[i:i+(f-1), j] = img3[i-1, j]
 
# Replicating columns
for i in range(0, m-1):
    for j in range(1, n-1, f):
        img3[i, j:j+(f-1)] = img3[i, j-1]
 
# Plot the up sampled image
print('Up Sampled Image:')
plt.imshow(img3, cmap="gray")


# In[5]:



#original image
import cv2
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("color1.Jpg")
plt.imshow(im)
plt.show()

#up sampling
import cv2
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("color1.Jpg")
im = im.resize((im.width*5, im.height*5), Image.NEAREST)
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.show()

#down sampling
im = Image.open("color1.Jpg")
im = im.resize((im.width//5, im.height//5))
plt.figure(figsize=(15,10))
plt.imshow(im)
plt.show()


# In[6]:


from PIL import Image,ImageDraw,ImageFilter
im1=Image.open('bird1.png')
im2=Image.open('bird2.png')
mask_im=Image.new("L",im2.size,0)
draw=ImageDraw.Draw(mask_im)
draw.ellipse((210,105,410,300),fill=225)
mask_im_blur=mask_im.filter(ImageFilter.GaussianBlur(10))
back_im=im1.copy()
back_im.paste(im2,(0,0),mask_im_blur)
back_im.show()


# In[21]:


#QUANTISATION

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

im = Image.open('color1.jpg')
pylab.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
 im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
 pylab.subplot(4,2,i), pylab.imshow(im1), pylab.axis('off')
 snr_list.append(signaltonoise(im1, axis=None))
 pylab.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
 str(np.round(snr_list[i-1],3)), size=20)
 i += 1
pylab.subplots_adjust(wspace=0.2, hspace=0)
pylab.show()


# In[23]:


from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

im = Image.open('bird2.png')
pylab.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
 im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
 pylab.subplot(4,2,i), pylab.imshow(im1), pylab.axis('off')
 snr_list.append(signaltonoise(im1, axis=None))
 pylab.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
 str(np.round(snr_list[i-1],3)), size=20)
 i += 1
pylab.subplots_adjust(wspace=0.2, hspace=0)
pylab.show()


# In[18]:




import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("color1.jpg")

res = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(res,cmap = 'gray')
plt.title('Downsampled Image')

plt.show()


# In[ ]:





# In[ ]:




