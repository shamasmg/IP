#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import the modules
import os
from os import listdir
 
# get the path or directory
folder_dir = "D:/DIPs"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")        or images.endswith(".jpeg") or images.endswith(".pdf") or images.endswith(".jfif")):
        # display
        print(images)


# In[11]:


import os
from os import listdir
 
# get the path or directory
folder_dir = "D:/DIPs"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")        or images.endswith(".jpeg")  or images.endswith(".jfif")):
        # display
        print(images)


# In[2]:


# import the modules
import os
from os import listdir
 
# get the path/directory
folder_dir = "D:/DIPs"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".png")):
        print(images)


# In[5]:


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
img_dir = "D:/DIPs" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    plt.figure()
    plt.imshow(img) 
    plt.show()


# In[6]:


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
img_dir = "D:/DIP2" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)
    plt.figure()
    plt.imshow(img) 
    plt.show()


# In[13]:


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
img_dir = "D:/DIP2" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0)
    data.append(img)
    plt.figure()
    plt.imshow(img, cmap="gray") 
    plt.show()


# In[10]:


import os

# List all files in a directory using os.listdir
basepath = 'D:/DIPs'
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        print(entry)


# In[27]:


import PIL
import os
import os.path
from PIL import Image

f = r'D:/DIP2'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((100,100))
    img.show(f_img)


# In[26]:


import PIL
import os
import os.path
from PIL import Image

f = r'D:/DIPs'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((100,100))
    img.show(f_img)


# In[24]:


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
img_dir = "D:/DIP2" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0)
    data.append(img)
    cv2.resize(img, (100,50))
    plt.figure()
    plt.imshow(img, cmap="gray") 
    plt.show()


# In[30]:


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
img_dir = "D:/DIP3/" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = plt.imread(f1)
    data.append(img)
    #image_resized=cv2.resize(img,(200,200))
    plt.figure()
    plt.title('Original Image')
    plt.imshow(img)
    #plt.show()
    #plt.imshow(image_resized)
    plt.show()
    #fig = plt.figure()
    #fig.set_size_inches(5, 5)
for f1 in files:
    img = plt.imread(f1)
    data.append(img)
    image_resized=cv2.resize(img,(200,200))
    plt.figure()
    plt.title('Resized Image')
    #plt.imshow(img)
    #plt.show()
    plt.imshow(image_resized)
    plt.show()


# In[ ]:




