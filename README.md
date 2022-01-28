
# Drowsiness Detection on Jetson Nano 2GB Developer Kit using Yolov5.

Drowsiness detection system which will detect whether a person is feeling sleepy
or not based on his/her behaviour of yawning while driving a vehicle.

## Aim and Objectives

#### Aim

To create a Drowsiness detection system which will detect whether a person is feeling sleepy
or not based on his/her behaviour of yawning while driving a vehicle.

#### Objectives
•
The main objective of the project is to create a program which can be either run on Jetson
nano or any pc with YOLOv5 installed and start detecting using the camera module on the
device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a person is feeling sleepy
or not.
## Abstract

• A person’s state of being i.e. whether he is feeling sleepy or not can be detected by the
live feed derived from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning
(ML), where machines are trained to identify various objects from one another. Machine
Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.

• A person should be completely awake while driving as a tired or sleepy person could
become a cause of dangerous mishaps.

• Lack of attention due to sleepiness or tiredness when driving a vehicle can cause accidents
and even endanger the life of not only the driver but also the passengers.
## Introduction


• This project is based on a Drowsiness detection model with modifications. We are going
to implement this project with Machine Learning and this project can be even run on jetson
nano which we have done.

• This project can also be used to gather information about a person’s usual behaviour when
driving, i.e., Sleepy, Awake.

• Person’s behaviour can be even classified further whether tired, sleepy, smiling, crying,
etc based on the image annotation we give in roboflow.

• Drowsiness detection or behaviour detection becomes difficult sometimes because of
various reasons like lighting inside the vehicle cabin, face in opposite direction of
viewfinder etc. However, training our model in roboflow has allowed us to compensate
for darkness as well as to crop the images according to what we see fit.

• Neural networks and machine learning have been used for these tasks and have obtained
good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Drowsiness detection as well.
## Literature Review

• Sleep plays a vital role in good health and well-being throughout your life. Getting enough
quality sleep at the right times can help protect your mental health, physical health, quality
of life, and safety.

• The way you feel while you're awake depends in part on what happens while you're
sleeping. During sleep, your body is working to support healthy brain function and
maintain your physical health. In children and teens, sleep also helps support growth and
development.

• Sleep helps your brain work properly. While you're sleeping, your brain is preparing for
the next day. It's forming new pathways to help you learn and remember information.

• Sleep plays an important role in your physical health. For example, sleep is involved in
healing and repair of your heart and blood vessels.Your immune system relies on sleep to
stay healthy. This system defends your body against foreign invasions.

• Studies show that a good night's sleep improves learning. Whether you're learning math,
how to play the piano, how to perfect your golf swing, or how to drive a car, sleep helps
enhance your learning and problem-solving skills. Sleep also helps you pay attention,
make decisions, and be creative.

• Sleepiness decreases our vigilance, response time, memory and decision making which
are all essential ingredients of safe driving. In fact, it is more detrimental to safety than
drunken driving.

• The injuries and deaths due to the road accidents impose a severe financial burden and
push the respective victims’ households into poverty and the already poor into debt.

• Road crashes also impact heavily on the country’s human assets. Around 76.2% of people
who are killed in road crashes in India are in their prime working-age, 18-45 years. This
means the country loses a massive number of its workforce every year, just because of
road accidents.
## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.

## Jetson Nano 2GB



https://user-images.githubusercontent.com/89011801/151481042-45cb8c9c-c61f-458a-a3f1-ed9dd1979326.mp4





## Methodology

The Drowsiness detection system is a program that focuses on implementing real time drowsiness
detection.

It is a prototype of a new product that comprises of the main module:
Drowsiness detection and then showing on viewfinder whether the person is sleepy or not.

#### Drowsiness Detection Module

#### This Module is divided into two parts:
#### 1] Drowsiness detection

• Ability to detect the location of face in any input image or frame. The output is the
bounding box coordinates on the detected face.

• For this task, initially the Dataset library Kaggle was considered. But integrating it
was a complex task so then we just downloaded the images from gettyimages.ae
and google images and made our own dataset.

• This Datasets identifies face in a Bitmap graphic object and returns the bounding
box image with annotation of face present in a given image.
    
#### 2] Behaviour Detection

• Classification of the face based on whether it is yawning or not.

• Hence YOLOv5 which is a model library from roboflow for image classification
and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use
in production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments is
straightforward.

• YOLOv5 was used to train and test our model for various classes like Drowsy,
Awake. We trained it for 149 epochs and achieved an accuracy of approximately
92%.
## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```

```bash
vim ~/.bashrc
```

```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```

## Drowsiness Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and pass them into yolov5 folder
link of project


## Running Drowsiness Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Demo




https://user-images.githubusercontent.com/89011801/151314733-0e9af0ae-5617-481a-acd7-8a20321506d4.mp4





## Advantages

• The Drowsiness detection system will be of great help to prevent accidents from
happening.

• The Drowsiness detection system can give either visual or audible cues to the driver when
it detects the drowsy behaviour pattern.

• It can also tell the the driver to take appropriate measures like washing face, do stretching,
drinking hot beverage,to get fresh air etc.

• As it is completely automatic it doesn’t require any user input and just works on the basis
of natural behaviour of the driver while driving.

• No one needs to keep an eye on the driver for fear of mishaps due to inadequate sleep of
driver.
## Application

• Detects Drowsy behaviour in each image frame or viewfinder using a camera module.

• Can be used in vehicles which travel for longer distances on a regular basis like truck
drivers or cargo ship captains etc.

• Can be used as a reference for other ai models based on Drowsiness detection.
## Future Scope

• As we know technology is marching towards automation, so this project is one of the
steps towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

• Detection of even small ticks for an individual like touching ear when feeling sleepy and
further customization of the model according to individual needs and behaviour like
feeling drowsy after heavy eating can be considered as an important step towards better
detection and prevention of accidents.
## Conclusion

• In this project our model is trying to detect yawning by a person and then showing it on
viewfinder, live as what the state of person is whether sleepy or awake.

• The model solves the problem of accidents and mishaps that occur due to unattentive or
sleepy drivers and makes roads safer for everyone else.

• Lesser accidents means lower death rate and hence lower hospital bills and also less harm
to country’s human assets which further improves the economy.
## Refrences

#### 1] Roboflow :- https://roboflow.com/

#### 2] Datasets or images used :- https://www.gettyimages.ae/photos/drowsy-driver?assettype=image&license=rf&alloweduse=availableforalluses&family=creative&phrase=drowsy%20driver&sort=mostpopular

#### 3] Google images
