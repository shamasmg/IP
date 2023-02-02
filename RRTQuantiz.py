#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Text on image using plot
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
img=Image.open("nature3.jpg")
d1=ImageDraw.Draw(img)
font=ImageFont.truetype('arial.ttf',50)
d1.text((500,500),"Hello how are you:",fill=(255,0,0),font=font)
plt.imshow(img)
plt.show()


# In[8]:


from PIL import Image
from PIL import ImageDraw
img = Image.open('nature3.jpg')
I1 = ImageDraw.Draw(img)
font = ImageFont.truetype("arial",50)
I1.text((28, 36),"GOOD MORNING", fill=(255, 0, 0),font=font)
img.show()
img.save("image.png")


# In[3]:


#resize

from PIL import Image
filepath='car2.jpg'
img=Image.open(filepath)
img.show()
new_image=img.resize((300,200))
new_image.show()


# In[4]:


#rotate

from PIL import Image
im=Image.open('flower3.jpg')
angle=45
out=im.rotate(angle,expand=True)
out.save('rotate_output.jpg')
out


# In[6]:


#text on image

from PIL import Image,ImageDraw,ImageFont
img=Image.open('nature3.jpg')
d1=ImageDraw.Draw(img)
font=ImageFont.truetype('arial.ttf',50)
d1.text((500,500),"Hello how are you ?",fill=(255,0,0),font=font)
img.show()


# In[7]:


#quantization

import cv2
from PIL import Image
image=Image.open('nature3.jpg')
image.show()
img=image.quantize(19)
img.show()


# In[ ]:




