#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
image=cv2.imread('car1.jpg')
cv2.imshow('Display image',image)
cv2.imwrite('D:\car1.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import cv2 
image=cv2.imread('car2.jpg')
cv2.imshow('Display image',image)
cv2.imwrite('D:\car2.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


down_width = 500
down_height = 320
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.destroyAllWindows()


# In[4]:


# importing cv2 module
import cv2

image_one = cv2.imread('car1.jpg')
image_two = cv2.imread('car2.jpg')

result_image = cv2.addWeighted(image_one, 0.5, image_two, 0.5, 0)

cv2.imshow('Final Image', result_image)
# deallocating the memory
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()


# In[5]:


import cv2

image_one = cv2.imread('car1.jpg')
image_two = cv2.imread('car2.jpg')

result_image = cv2.subtract(image_one,image_two)

cv2.imshow('Final Image', result_image)
# deallocating the memory
if cv2.waitKey(0) & 0xff == 27:
   cv2.destroyAllWindows()


# In[6]:


import cv2
img = cv2.imread('car3.jpg')
 
# Applying OpenCV scalar multiplication on image
fimg = cv2.multiply(img, 1.5)
 
# Saving the output image
cv2.imshow('output.jpg', fimg)


# In[7]:




import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('car1.jpg')
imge2=plt.imread('car2.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1+imge2
plt.imshow(addimg)
plt.show()


# In[8]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('car1.jpg')
imge2=plt.imread('car2.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1-imge2
plt.imshow(addimg)
plt.show()


# In[9]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('car1.jpg')
imge2=plt.imread('car2.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1*imge2
plt.imshow(addimg)
plt.show()


# In[10]:


import cv2
import matplotlib.pyplot as plt
imge1=plt.imread('car1.jpg')
imge2=plt.imread('car2.jpg')
plt.imshow(imge1)
plt.show()
plt.imshow(imge2)
plt.show()
addimg=imge1/imge2
plt.imshow(addimg)
plt.show()


# In[11]:


from PIL import Image
img = Image.open('bfly3.png')
img.save("D:/Butterfly.tiff",'TIFF')




# In[12]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.camera()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[13]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.clock()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[14]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.coins()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[15]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.horse() 
plt.imshow(image) 

type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
 


# In[16]:


import numpy as np
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
image =data.horse()  
viewer = ImageViewer(image)
viewer.show()


# In[17]:


from PIL import Image

# Function to change the image size
def changeImageSize(maxWidth,
                    maxHeight,
                    image):
   
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage
   
# Take two images for blending them together  
image1 = Image.open("car3.jpg")
image2 = Image.open("flower1.png")

# Make the images of uniform size
image3 = changeImageSize(800, 500, image1)
image4 = changeImageSize(800, 500, image2)

# Make sure images got an alpha channel
image5 = image3.convert("RGBA")
image6 = image4.convert("RGBA")

# Display the images
image5.show()
image6.show()

# alpha-blend the images with varying values of alpha
alphaBlended1 = Image.blend(image5, image6, alpha=.2)
alphaBlended2 = Image.blend(image5, image6, alpha=.4)

# Display the alpha-blended images
alphaBlended1.show()
alphaBlended2.show()


# In[ ]:





# In[18]:


import cv2
image = cv2.imread("car3.jpg")

y=0
x=0
h=600
w=810
crop_image = image[x:w, y:h]
cv2.imshow("Cropped", crop_image)
cv2.waitKey(0)


# In[19]:


from PIL import Image

#Create an Image Object from an Image
im = Image.open('D:/DIPs/bfly3.png')

#Display actual image
im.show()

#left, upper, right, lowe
#Crop
cropped = im.crop((1,2,300,300))

#Display the cropped portion
cropped.show()

#Save the cropped image
cropped.save('D:/bflycrop.png')


# In[20]:


from PIL import Image, ImageOps

im = Image.open('D:/DIPs/car2.jpg')
im_invert = ImageOps.invert(im)
im_invert.save('D:/images/carneg.jpg', quality=95)
im_invert.show()


# In[ ]:





# In[21]:


import numpy as np
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
image =data.horse()  
viewer = ImageViewer(image)
viewer.show()


# In[ ]:





# In[ ]:





# In[ ]:




