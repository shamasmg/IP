#!/usr/bin/env python
# coding: utf-8

# In[5]:


#face,eyes and mouth detection
import cv2

#load the xml files for face, eye and mouth detection into the program
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth1.xml')
#read the image for furthur editing
image = cv2.imread('woman.jpg')
#show the original image
cv2.imshow('Original image', image)
cv2.waitKey(100)
#convert the RBG image to gray scale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#identify the face using haar-based classifiers
faces = face_cascade.detectMultiScale(image, 1.4, 4)
#iteration through the faces array and draw a rectangle
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    #identify the eyes and mouth using haar-based classifiers
    eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray_image, 1.5, 11)
    #iteration through the eyes and mouth array and draw a rectangl
    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(image,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for(mx, my, mw, mh) in mouth:
            cv2.rectangle(image, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
            #show the final image after detection
            cv2.imshow('face, eyes and mouth detected image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #show a successful message to the user
            #print("Face, eye and mouth detection is successful")


# In[6]:


#smile detection
import cv2

# read input image
img = cv2.imread('woman.jpg')

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#read haar cascade for smile detection
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('Number of detected faces:', len(faces))

# loop over all the faces detected
for (x,y,w,h) in faces:
   
   # draw a rectangle in a face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
 
   # detecting smile within the face roi
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    if len(smiles) > 0:
        print("smile detected")
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            cv2.putText(roi_color, "smile", (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
                print("smile not detected")
                # Display an image in a window
                cv2.imshow('Smile Image',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# In[8]:


#nose detection
import cv2

# read input image
img = cv2.imread('woman.jpg')

# convert to grayscale of each frames
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#read haar cascade for smile detection
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Detects faces in the input image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print('Number of detected faces:', len(faces))

# loop over all the faces detected
for (x,y,w,h) in faces:
   
   # draw a rectangle in a face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
 
   # detecting smile within the face roi
    nose = nose_cascade.detectMultiScale(roi_gray, 1.8, 20)
    if len(nose) > 0:
        print("nose detected")
        for (sx, sy, sw, sh) in nose:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            cv2.putText(roi_color, "nose", (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
                print("nose not detected")
                # Display an image in a window
                cv2.imshow('nose Image',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# In[7]:


#face,eyes,nose and mouth detection
import cv2

#load the xml files for face, eye and mouth detection into the program
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth1.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
#read the image for furthur editing
image = cv2.imread('woman.jpg')
#show the original image
cv2.imshow('Original image', image)
cv2.waitKey(100)
#convert the RBG image to gray scale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#identify the face using haar-based classifiers
faces = face_cascade.detectMultiScale(image, 1.4, 4)
#iteration through the faces array and draw a rectangle
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    #identify the eyes and mouth using haar-based classifiers
    eyes = eye_cascade.detectMultiScale(gray_image, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray_image, 1.5, 11)
    nose_rects = nose_cascade.detectMultiScale(gray_image, 1.3, 5)
    #iteration through the eyes and mouth array and draw a rectangl
    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(image,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        for(mx, my, mw, mh) in mouth:
            cv2.rectangle(image, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
            for(nx, ny, nw, nh) in  nose_rects:
                cv2.rectangle(image, (nx,ny), (nx+nw,ny+nh), (0,255,0), 3)
            #show the final image after detection
            cv2.imshow('face, eyes,nose and mouth detected image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #show a successful message to the user
            #print("Face, eye and mouth detection is successful")


# In[ ]:




