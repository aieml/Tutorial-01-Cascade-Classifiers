# Tutorial-02
In this tutorial you will learn about cascade classifiers a technology introduced in early 2000's for object detection. At the end of the tutorial you will have a assignment based on Cascade Classifiers. 

## Introduction
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

[Read more @ opencv official documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

## How to detect objects using cascade classifiers

This repository contains two pretrained cascade classifers for detecting face and eyes. You can download more cascade classifiers from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)

```python
dataset=pd.read_csv('heart.csv').as_matrix()
```
