#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Logo removed
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
 
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
       

def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')


image_with_logo = plt.imread('logo.png')

# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[210:290, 360:425] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')


# In[6]:


#Logo removed
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
from PIL import Image
from skimage import io
import cv2

image_with_logo = plt.imread('logo.png')
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
       

def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    #ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    #ax2.axis('off')




# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[210:290, 360:425] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')


# In[2]:


import numpy as np
import cv2
 
# Open the image.
img = cv2.imread('dm.png')
 
# Load the mask.
mask = cv2.imread('cat_msk.png', 0)
 
# Inpaint.
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
 
# Write the output.
cv2.imwrite('cat_inpainted.png', dst)
cv2.imshow('cat_inpainted.png', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np
from skimage import io      # Only needed for web grabbing images; use cv2.imread(...) for local images

# Read images
frame = cv2.cvtColor(io.imread('crop.png'), cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(io.imread('nature3.jpg'), cv2.COLOR_RGB2BGR)

# Color threshold red frame; single color here, more sophisticated solution would be using cv2.inRange
mask = 255 * np.uint8(np.all(frame == [36, 28, 237], axis=2))

# Find inner contour of frame; get coordinates
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = min(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(cnt)

# Copy appropriately resized image to frame
frame[y:y+h, x:x+w] = cv2.resize(image, (w, h))

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


import cv2
import numpy as np
from skimage import io      # Only needed for web grabbing images; use cv2.imread(...) for local images

# Read images
frame = cv2.cvtColor(io.imread('crop.png'), cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(io.imread('nature.jpg'), cv2.COLOR_RGB2BGR)

# Color threshold red frame; single color here, more sophisticated solution would be using cv2.inRange
mask = 255 * np.uint8(np.all(frame == [36, 28, 237], axis=2))

# Find inner contour of frame; get coordinates
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = min(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(cnt)

# Copy appropriately resized image to frame
frame[y:y+h, x:x+w] = cv2.resize(image, (w, h))

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
    
def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')


image_with_logo = plt.imread('logo2.webp')

# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[173:280, 255:355] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')


# In[6]:


import cv2
im = cv2.imread("logo2.webp", cv2.IMREAD_UNCHANGED)
_, mask = cv2.threshold(im[:, :, 2], 300, 400, cv2.THRESH_BINARY)
cv2.imwrite('mask.jpg', mask)
cv2.imshow('mask.jpg', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


import cv2
im = cv2.imread("dm.png", cv2.IMREAD_UNCHANGED)
_, mask = cv2.threshold(im[:,:, 2], 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('mask.jpg', mask)
cv2.imshow('mask.jpg', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


# REMOVING TEXT FROM IMAGE

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
def inpaint_text(img_path, pipeline):
    # read the image
    img = keras_ocr.tools.read(img_path)
   
     
    prediction_groups = pipeline.recognize([img])
   

    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
       
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
       
   
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
       
       
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)
pipeline = keras_ocr.pipeline.Pipeline()

img_text_removed = inpaint_text('cat.png', pipeline)

plt.imshow(img_text_removed)

cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))


# In[9]:


#Text removal
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
def inpaint_text(img_path, pipeline):
    # read the image
    img = keras_ocr.tools.read(img_path)
   
     
    prediction_groups = pipeline.recognize([img])
   

    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]
       
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
       
   
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
       
       
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)
pipeline = keras_ocr.pipeline.Pipeline()

img_text_removed = inpaint_text('watermark.jpg', pipeline)

plt.imshow(img_text_removed)

cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))


# In[24]:


import numpy as np
import cv2 
img = cv2.imread('nature.jpg')

print(img.shape)

h,w,c =  img.shape

logo = cv2.imread("crop2.png")
print(logo.shape)
hl,wl,cl  =  logo.shape

x1 = int(w/2-wl/2)
y1 = int(h/2-hl)
x2 = int(w/2+wl/2)
y2 =  int(h/2)
cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

#cv2.imwrite("my.png",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[50]:


import numpy as np
import cv2 
img = cv2.imread('apple.jpg')

print(img.shape)

h,w,c =  img.shape

logo = cv2.imread("apple3.jpg")
print(logo.shape)
hl,wl,cl  =  logo.shape

x1 = int(w/2-wl/2)
y1 = int(h/2-hl)
x2 = int(w/2+wl/2)
y2 =  int(h/2)
cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

#cv2.imwrite("my.png",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[29]:


import numpy as np
import cv2 
img = cv2.imread('nature.jpg')

print(img.shape)

h,w,c =  img.shape

logo = cv2.imread("ncrop2.jpg")
print(logo.shape)
hl,wl,cl  =  logo.shape

x1 = int(w/2-wl/2)
y1 = int(h/2-hl)
x2 = int(w/2+wl/2)
y2 =  int(h/2)
cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)

#cv2.imwrite("my.png",img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[1]:


from PIL import Image
img=Image.open("nature.jpg")
b=(100,120,250,250)
c_i=img.crop(box=b)
c_i.show()
c_i.save('D:/ncrop1.jpg')


# In[46]:


from PIL import Image
img=Image.open("apple.jpg")
b=(250,310,320,410)
c_i=img.crop(box=b)
c_i.show()
c_i.save('D:/DIPs/apple1.jpg')


# In[49]:


import cv2
im = cv2.imread("apple1.jpg", cv2.IMREAD_UNCHANGED)
_, mask = cv2.threshold(im[:, :, 2], 100, 255, cv2.THRESH_BINARY)
cv2.imshow('apple3.jpg', mask)
cv2.imwrite('apple3.jpg', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


import cv2
im = cv2.imread("crop1.png", cv2.IMREAD_UNCHANGED)
_, mask = cv2.threshold(im[:, :, 2], 100, 255, cv2.THRESH_BINARY)
cv2.imshow('crop2.png', mask)
cv2.imwrite('crop2.png', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


import cv2
im = cv2.imread("ncrop1.jpg", cv2.IMREAD_UNCHANGED)
_, mask = cv2.threshold(im[:, :, 2], 100, 255, cv2.THRESH_BINARY)
cv2.imshow('ncrop2.jpg', mask)
#cv2.imwrite('ncrop2.jpg', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


from PIL import Image
img=Image.open("logo3.png")
b=(600,100,800,200)
c_i=img.crop(box=b)
c_i.show()
#c_i.save('D:/crop13.png')


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
    
def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')


image_with_logo = plt.imread('nature.jpg')

# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[100:250,120:250] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')


# In[28]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
    
def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')


image_with_logo = plt.imread('logo3.png')

# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[500:700,100:200] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo,
                                                mask,
                                                multichannel=True)

# Show the original and logo removed images
plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')


# In[16]:


#Adding noise
import matplotlib.pyplot as plt
from skimage.util import random_noise

def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')

bfly_img=plt.imread("bfly.jfif")

#Add noise to the image
noisy_img=random_noise(bfly_img)

#show the original and resulting image
plot_comparison(bfly_img,noisy_img,"noisy image")


# In[11]:


#Redusing noise
from skimage.restoration import denoise_tv_chambolle

noisy_image=plt.imread('noisy2.png')

#Apply total variation filter denoising
denoised_image=denoise_tv_chambolle(noisy_image,multichannel=True)

#Show the noisy and denoised image
plot_comparison(noisy_image,denoised_image,"denoised image")


# In[12]:


#Segmntation
from skimage.segmentation import slic
from skimage.color import label2rgb

img=plt.imread('mask2.png')

#obtain the segmentation with 400 regions
segments=slic(img,n_segments=400)

#put segments on top of original image to compare
segmented_image=label2rgb(segments,img,kind='avg')

#Show the segmented image
plot_comparison(img,segmented_image,'SEgmented image,400 superpixels')


# In[15]:


#Segmntation
from skimage.segmentation import slic
from skimage.color import label2rgb

img=plt.imread('car2.jpg')

#obtain the segmentation with 400 regions
segments=slic(img,n_segments=400)

#put segments on top of original image to compare
segmented_image=label2rgb(segments,img,kind='avg')

#Show the segmented image
plot_comparison(img,segmented_image,'SEgmented image,400 superpixels')


# In[ ]:




