#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import cv2
import imageio
from imageio import imread
im = rgb2gray(imread("sci.jpg")) # read the image from disk as a numpy ndarray
plt.figure(figsize=(20,8))
plt.subplot(131), plt.imshow(im, cmap='gray'), plt.title('Original Image',size=20)
plt.subplot(132), plt.contour(np.flipud(im), colors='k',levels=np.logspace(-15, 15, 100))
plt.title('Image Contour Lines', size=20)
plt.subplot(133), plt.title('Image Filled Contour',size=20),
plt.contourf(np.flipud(im), cmap='inferno')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import cv2
import imageio
from imageio import imread
im = rgb2gray(imread("bfly2.jfif")) # read the image from disk as a numpy ndarray
plt.figure(figsize=(20,8))
plt.subplot(131), plt.imshow(im, cmap='gray'), plt.title('Original Image',size=20)
plt.subplot(132), plt.contour(np.flipud(im), colors='k',levels=np.logspace(-15, 15, 100))
plt.title('Image Contour Lines', size=20)
plt.subplot(133), plt.title('Image Filled Contour',size=20),
plt.contourf(np.flipud(im), cmap='inferno')
plt.show()


# In[20]:


import matplotlib.pyplot as plt
def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation='nearest', cmap='gray_r')
    plt.title('contours')
    plt.axis('off')

from skimage import measure, data
horse_image = data.horse()
contours = measure.find_contours(horse_image, level=0.8)
show_image_contour(horse_image, contours)
   


# In[1]:


#find contours of an image that is not binary
from skimage.io import imread
from skimage.filters import threshold_otsu

image_dices=imread('')

#Make the image grayscale
image_dices=color.rgb2gray(image_dices)

#Obtain the optimal thresh value
thresh=threshold_otsu(image_dices)

#Apply thresholding
binary=image_dices>thresh

#find contours at a constant value of 0.8
contours=measure.find_contours(binary,level=0.8)

#show the image
show_image_contour(image_dices,contours)


# In[4]:


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

images = []
for img_path in glob.glob('D:/DIP2/*.jpg'):
    images.append(mpimg.imread(img_path))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)


# In[45]:


from PIL import Image, ImageDraw

collage = Image.new("RGBA", (1500,1500), color=(255,255,255,255))
lst = [5]

c=0
for i in range(0,1500,500):
    for j in range(0,1500,500):
        file = "flower3.jpg"
        
        photo = Image.open(file).convert("RGBA")
        photo = photo.resize((500,500))        
        
        collage.paste(photo, (i,j))
        c+=1
collage.show()


# In[49]:


import cv2
from PIL import Image
from skimage import io

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

def create_collage(images):
    images = [io.imread(img) for img in images]
    images = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)) for image in images]
    if len(images) > 2:
        half = len(images) // 2
        h1 = cv2.hconcat(images[:half])
        h2 = cv2.hconcat(images[half:])
        concat_images = cv2.vconcat([h1, h2])
    else:
        concat_images = cv2.hconcat(images)
    image = Image.fromarray(concat_images)

    # Image path
    image_name = "kk.png"
    image = image.convert("RGB")
    image.save(f"{image_name}")
    return image_name
images=["car3.jpg","car2.jpg","flower3.jpg","color1.jpg"]
#image1 on top left, image2 on top right, image3 on bottom left,image4 on bottom right
create_collage(images)
#cv2.imshow(' ',create_collage)


# In[5]:


import cv2
from PIL import Image
from skimage import io

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

def create_collage(images):
    images = [io.imread(img) for img in images]
    images = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)) for image in images]
    if len(images) > 2:
        half = len(images) // 2
        h1 = cv2.hconcat(images[:half])
        h2 = cv2.hconcat(images[half:])
        concat_images = cv2.vconcat([h1, h2])
    else:
        concat_images = cv2.hconcat(images)
    image = Image.fromarray(concat_images)

    # Image path
    image_name = "kk1.png"
    image = image.convert("RGB")
    image.save(f"{image_name}")
    return image_name
images=["bird1.png","bird2.png","flower1.png","flower2.png"]
#image1 on top left, image2 on top right, image3 on bottom left,image4 on bottom right
create_collage(images)
#cv2.imshow(' ',create_collage)


# In[ ]:




