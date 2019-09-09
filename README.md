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
#loading the face detecting cascade classifier into face_clsfr
eye_clsfr=cv2.CascadeClassifier('Cascades\haarcascade_eye_tree_eyeglasses.xml')
#loading the eyes cascade classifier into face_clsfr

camera=cv2.VideoCapture(0)
#initializing the video object (0 for default webcam)

while(True):
#infinite loop to read continuous frames from the camera object

    ret,img=camera.read()
    #reading a single frame from the camera
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #converting the color image to a gray scale image
    faces=face_clsfr.detectMultiScale(gray)     
    #detecting faces in the gray scale, (this is quite similar to results=clsfr.predict(data))
    #faces is a 2D array contaning n number of rows (n= number of faces in the frame), 4 columns (x,y,w,h)
    for (x,y,w,h) in faces:
    #going through each and every face and assigning the x,y,w,h

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #Drawing a rectangle bounding the faces
        cv2.putText(img,'FACE',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        #showing 'FACE' near the bounding rectangle

        face_img=gray[y:y+w,x:x+w]
        #cropping the detected face area and assigning into a face_img 
        #cv2.imshow('Face',face_img)
        #comment the above line to see the cropped face
        eyes=eye_clsfr.detectMultiScale(face_img)
        #detecting eyes in the cropped face

        for(ex,ey,ew,eh) in eyes:

            cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
            cv2.putText(img,'EYES',(x+ex,y+ey-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        #drawing rectangles bounding eyes
            
    cv2.imshow('LIVE',img)
    cv2.waitKey(1)
    #showing the frame
```

### x,y,w,h in the image

<img src="https://miro.medium.com/max/1360/0*De1DLB3Io5DAzfWl." width="400">
