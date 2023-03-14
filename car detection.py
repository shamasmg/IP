#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open(requests.get('https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg', stream=True).raw)
image = image.resize((450,250))
image_arr = np.array(image)
image


# In[2]:


grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)


# In[3]:


blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)


# In[4]:


dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)


# In[5]:


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)


# In[7]:


car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cars


# In[8]:


cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)


# In[53]:


# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open('car10.jpg')
image = image.resize((450,250))
image_arr = np.array(image)
image


# In[54]:


grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)


# In[55]:


blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)


# In[56]:


dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)


# In[57]:


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)


# In[58]:


car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cars


# In[59]:


cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)


# In[18]:


# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open('car2.jpg')
image = image.resize((450,250))
image_arr = np.array(image)
image


# In[20]:


dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cars

cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)


# In[33]:


# Import libraries
from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open('car7.jpg')
image = image.resize((450,250))
image_arr = np.array(image)
image


# In[34]:


dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
Image.fromarray(closing)

car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cars

cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found")
Image.fromarray(image_arr)


# In[ ]:




