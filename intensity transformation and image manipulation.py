#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('color1.jpg')
plt.figure(figsize=(6,6))
plt.imshow(pic);
plt.axis('off');


# In[7]:


negative = 255-pic 
plt.figure(figsize=(6,6))
plt.imshow(negative);
plt.axis('off');


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic=imageio.imread('flower3.jpg')
plt.figure(figsize=(6,6))
plt.imshow(pic);
plt.axis('off');


# In[15]:


#image negative
negative = 255-pic #neg=(L-1)-img 
plt.figure(figsize=(6,6))
plt.imshow(negative);
plt.axis('off');


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import numpy as np
import matplotlib.pyplot as plt
 
pic=imageio.imread('flower3.jpg')
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)

max_=np.max(gray)

def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)
plt.figure(figsize=(5,5))
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
plt.axis('off');


# In[18]:


import imageio
import matplotlib.pyplot as plt

#Gamma encoding
pic=imageio.imread('flower3.jpg')
gamma=2.2 #Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright

gamma_correction=((pic/255)**(1/gamma))
plt.figure(figsize=(5,5))
plt.imshow(gamma_correction)
plt.axis('off');


# # Image manipulation

# In[20]:


#Image sharpeness
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
#Load the image
my_image=Image.open('flower2.png')
# Use sharpen function
sharp=my_image.filter(ImageFilter.SHARPEN)
# save the image
sharp.save('D:/image_sharpen.png')
sharp.show()
plt.imshow(sharp)
plt.show()


# In[21]:


#Image flip
import matplotlib.pyplot as plt
#Load the image
img=Image.open('flower2.png')
plt.imshow(img)
plt.show()
# Use the flip function
flip=img.transpose(Image.FLIP_LEFT_RIGHT)

# save the image
flip.save("D:/image_flip.png")
plt.imshow(flip)
plt.show()


# In[30]:


#Importing Image class from PIL module
from PIL import Image
import matplotlib.pyplot as plt
# opens a image in RGB mode
im=Image.open('flower3.jpg')

#size of the image in pixels (size of original image)
# (This is not mandatory)
width,height=im.size

#cropped image of above dimension
# (It will not change original image)
im1=im.crop((280,100,800,600))

#shows the image in image viewer
im1.show()
plt.imshow(im1)
plt.show()


# In[1]:



#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 10))

# setting values to rows and column variables
rows = 4
columns = 2

# reading images
Image1 = cv2.imread('color1.jpg')
Image2 = cv2.imread('car3.jpg')
Image3 = cv2.imread('car2.jpg')
Image4 = cv2.imread('flower3.jpg')

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("First")

fig.add_subplot(rows, columns, 2) 

negative = 255-Image1 
#plt.figure(figsize=(6,6))
plt.imshow(negative)
plt.axis('off')
plt.title("Negative")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("second")

fig.add_subplot(rows, columns, 4)

negative2 = 255-Image2
#plt.figure(figsize=(6,6))
plt.imshow(negative2);
plt.axis('off');
plt.title("Negative")

 # Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("third")

fig.add_subplot(rows, columns, 6)

negative3 = 255-Image3
#plt.figure(figsize=(6,6))
plt.imshow(negative3);
plt.axis('off');
plt.title("Negative")   
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 7)

# showing image
plt.imshow(Image4)
plt.axis('off')
plt.title("fourth")

fig.add_subplot(rows, columns, 8)

negative4 = 255-Image4
#plt.figure(figsize=(6,6))
plt.imshow(negative4);
plt.axis('off');
plt.title("Negative")


# In[81]:


#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 10))
  
# setting values to rows and column variables
rows = 3
columns = 3
  
# reading images
Image1 = cv2.imread('color1.jpg')
Image2 = cv2.imread('car3.jpg')
Image3 = cv2.imread('car2.jpg')
Image4 = cv2.imread('flower3.jpg')
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("Original image")

fig.add_subplot(rows, columns, 2) 

negative = 255-Image1 
#plt.figure(figsize=(6,6))
plt.imshow(negative)
plt.axis('off')
plt.title("Negative")

#Image sharpeness
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
my_image=Image.open('color1.jpg')

sharp=my_image.filter(ImageFilter.SHARPEN)

fig.add_subplot(rows, columns, 3) 

# save the image
sharp.save('D:/image_sharpen.png')
sharp.show()
plt.imshow(sharp)
plt.axis('off')
plt.title("sharpen")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(Image2)
plt.axis('off')
plt.title("Original image")

fig.add_subplot(rows, columns, 5)

negative2 = 255-Image2
#plt.figure(figsize=(6,6))
plt.imshow(negative2);
plt.axis('off');
plt.title("negative")
  


#Image sharpeness
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
my_image=Image.open('car3.jpg')

sharp=my_image.filter(ImageFilter.SHARPEN)

fig.add_subplot(rows, columns, 6) 

# save the image

sharp.show()
plt.imshow(sharp)
plt.axis('off')
plt.title("sharpen")


# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 7)
  
# showing image
plt.imshow(Image3)
plt.axis('off')
plt.title("Original image")

fig.add_subplot(rows, columns, 8)

negative4 = 255-Image3
#plt.figure(figsize=(6,6))
plt.imshow(negative4);
plt.axis('off');
plt.title("negative")

#Image sharpeness
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
my_image=Image.open('car2.jpg')

sharp=my_image.filter(ImageFilter.SHARPEN)

fig.add_subplot(rows, columns, 9) 

# save the image

sharp.show()
plt.imshow(sharp)
plt.axis('off')
plt.title("sharpen")


# In[3]:



  
#import necessary packages
import cv2
import imageio
import os
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
 
#Set the path where images are stored
img_dir = "D:/DIPs/" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    plt.figure()
    plt.imshow(img)
    rows,cols=6,2
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        negative=255-img #neg=(L-1)-img
        plt.figure(figsize=(6,6))
        plt.imshow(negative)
        plt.axis('off')
   
        plt.subplot(rows,cols,1)
   
        plt.imshow(img)
        plt.title("original image")

        plt.subplot(rows,cols,2)
   
        plt.imshow(negative)
        plt.title("negative image") 


# In[4]:


#import necessary packages
import cv2
import imageio
import os
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
 
#Set the path where images are stored
img_dir = "D:/DIP2/" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    plt.figure()
    plt.imshow(img)
    rows,cols=6,2
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        negative=255-img #neg=(L-1)-img
        plt.figure(figsize=(6,6))
        plt.imshow(negative)
        plt.axis('off')
   
        plt.subplot(rows,cols,1)
   
        plt.imshow(img)
        plt.title("original image")

        plt.subplot(rows,cols,2)
   
        plt.imshow(negative)
        plt.title("negative image") 


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import imageio
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
pic1=imageio.imread('car2.jpg')
plt.figure(figsize=(6,6))
plt.imshow(pic1);
pic2=imageio.imread('bfly2.jfif')
plt.figure(figsize=(6,6))
plt.imshow(pic2);
plt.axis('off');


negative1=255-pic1 #neg=(L-1) - img
plt.figure(figsize=(6,6))
plt.imshow(negative1);
negative2=255-pic2
plt.figure(figsize=(6,6))
plt.imshow(negative2);
plt.axis('off');


##plotting

rows, cols = 2,2
plt.subplot(rows, cols, 1)
plt.imshow(pic1)
plt.title("Original image")
plt.subplot(rows, cols, 2)
plt.imshow(negative1)
plt.title("Negative image")
plt.subplot(rows, cols, 3)
plt.imshow(pic2)
plt.title("Original image")
plt.subplot(rows, cols, 4)
plt.imshow(negative2)
plt.title("Negative image")

plt.show()


# In[ ]:





# In[ ]:




