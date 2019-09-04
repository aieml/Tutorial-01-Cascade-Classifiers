# Tutorial-02
In this tutorial you will learn about cascade classifiers a technology introduced in early 2000's for object detection. At the end of the tutorial you will have a assignment based on Cascade Classifiers. 

## Introduction
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

[Read more @ opencv official documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

## How to detect objects using cascade classifiers

This repository contains two pretrained cascade classifers for detecting face and eyes. You can download more cascade classifiers from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)

```python
import cv2

face_clsfr=cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
#loading the cascade classi
eye_clsfr=cv2.CascadeClassifier('Cascades\haarcascade_eye_tree_eyeglasses.xml')

camera=cv2.VideoCapture(0)

while(True):

    ret,img=camera.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray)     #results=clsfr.predict(features)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,'FACE',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        face_img=gray[y:y+w,x:x+w]
        #cv2.imshow('Face',face_img)
        eyes=eye_clsfr.detectMultiScale(face_img)

        for(ex,ey,ew,eh) in eyes:

            cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
            cv2.putText(img,'EYES',(x+ex,y+ey-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

            
            
    cv2.imshow('LIVE',img)
    cv2.waitKey(1)
```
