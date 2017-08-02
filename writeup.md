# Behavioral Cloning
---

## Table of Content
* [Introduction](#introduction)
* [Rubric Points](#rubric-points)
  * [Files Submitted &amp; Code Quality](#files-submitted--code-quality)
    * [1\. Submission includes all required files and can be used to run the simulator in autonomous mode](#1-submission-includes-all-required-files-and-can-be-used-to-run-the-simulator-in-autonomous-mode)
    * [2\. Submission includes functional code](#2-submission-includes-functional-code)
    * [3\. Submission code is usable and readable](#3-submission-code-is-usable-and-readable)
  * [Model Architecture and Training Strategy](#model-architecture-and-training-strategy)
    * [1\. An appropriate model architecture has been employed](#1-an-appropriate-model-architecture-has-been-employed)
    * [2\. Attempts to reduce overfitting in the model](#2-attempts-to-reduce-overfitting-in-the-model)
    * [3\. Model parameter tuning](#3-model-parameter-tuning)
    * [4\. Appropriate training data](#4-appropriate-training-data)
  * [Model Architecture and Training Strategy](#model-architecture-and-training-strategy-1)
    * [1\. Solution Design Approach](#1-solution-design-approach)
    * [2\. Final Model Architecture](#2-final-model-architecture)
    * [3\. Creation of the Training Set &amp; Training Process](#3-creation-of-the-training-set--training-process)

## Introduction

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]:  # (Image References)

[gif1]: ./gif1.gif
[gif2]: ./gif2.gif
[gif3]: ./gif3.gif

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modified version of the NVIDIA model introduced in the lecture.
In addition, a simpler network following LeNet-5 and a much more complex network based on fine-tuned Inception-V3 were also experimented.

The data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
A decay factor and early-stopping callback are used.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road as well as focused training on
segments of the road where accidents tend to occur in early iterations.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a few well-know architecture that performed well on image classification.

My first step was to use LeNet-5. Then I tried the NVIDIA model introduced in the lecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.
This implied that the model was overfitting. To combat the overfitting,
I modified the model to include dropout layers between all the fully connected layers with 0.5 drop probability.

The final step was to run the simulator to see how well the car was driving around track one.
There were a few spots where the vehicle fell off the track, most notably at the sharp turn leading into the bridge, on the bridge,
at the sharp turn after the bridge involving a dirt road, and at the turn following that.
To improve the driving behavior in these cases, I collect several training sessions just on these segments. In addition, I recorded recovery drive
in order to train the network to make better and bolder turns when the car is near the edge of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model (dubbed ```nvidia``` in the code) is described in the table below.

| Layer Type      		|     Description	|
|:---:|:---|
| Input and Normalization | 160x320x3 RGB image, divided by 255, then centered at zero |
| Cropping         		| Remove top 70 rows and bottom 25 rows |
| Convolution 5x5x24  	| 2x2 stride, ReLu activation |
| Convolution 5x5x36  	| 2x2 stride, ReLu activation |
| Convolution 5x5x48  	| 2x2 stride, ReLu activation |
| Convolution 3x3x64  	| 1x1 stride, ReLu activation |
| Convolution 3x3x64  	| 1x1 stride, ReLu activation |
| Flatten               | |
| Dropout               | Dropout 50% |
| Fully connected		| Outputs 1x100|
| Dropout               | Dropout 50% |
| Fully connected		| Outputs 1x50|
| Dropout               | Dropout 50% |
| Fully connected		| Outputs 1x10|
| Fully connected		| Outputs 1x1|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.
Here is an example image of center lane driving:

![alt text][gif3]

After a few iterations of training and simulations, it appeared the vehicle is prune to go off track near a few turn points and on the bridge.
So I acquired more training data focused on those segments of the lap with high accidents rate. The following shows the segment from the bridge
to the turn with a dirt road.

![alt text][gif2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center
so that the vehicle would learn to recover from mistakes and make bolder steering when the car is dangerously close to the edges.
The following gif shows what a recovery looks like:

![alt text][gif1]


To augment the data set, I also flipped images and angles thinking that this would work similarly as driving reversely on the track.
Additional augmentation is possible by making use of the left/right camera images, though in practice it didn't prove to improve results.

After the collection process, I had 33462 data points. For each images, I converted the image from BGR mode (due to OpenCV convention)
to RGB mode. I then preprocessed this data by normalizing, centering and cropping as mentioned in the model description above.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model.
The validation set helped determine if the model was over or under fitting.
I used early stopping callback let the training stop automatically when the validation loss no longer improves within 3 epochs.
For training model ````nvidia```` I used a batch size of 256 for training ````inception```` I used a batch size of 32.

The model was able to successfully drive around the track. One lap is recorded as [video.mp4](video.mp4)
