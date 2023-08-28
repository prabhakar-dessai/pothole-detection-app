# Pothole and Lane Detection App

## Introduction

The app aims to showcase a system that accurately detects obstacles and lane  lines using video input from dashboard of a  car by leveraging the RTMdet model of [mmdetection library](https://github.com/open-mmlab/mmdetection) for object detection and lane detection using computer vison techniques. 

The idea was inspired by the fact that most existing road datasets are mainly recorded in Western countries and do not account for challenges like waterlogged potholes and road defects commonly faced in India hence we recorded our own dataset targeting roads in the state of goa, India.The model is trained on a manually collected Goan road dataset spanning over 130KMs and additional online datasets. 

Another feature is that instead of relying on expensive sensors like Lidar and Radar, we have demostrated how object detection in autonomous vehicles can be implemented more affordable by leveraging Deep Learning models and Computer Vision techniques.

![landing_section](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/cf016a4f-9c31-4d58-94a8-f3d3096db3a4)
![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/173354c7-6528-4778-a358-f61865203c8c)
![demo_uploaded_images](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/d3c2176d-f867-4928-87bb-7b3bcc566742)

https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/f3ed844b-904b-44c6-8225-2a387d28feaf




![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/cd14d7cf-a3ac-4481-ba32-41df638feb44)




## Basic Flow of Model

  
![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/cdf03816-c8b8-41bf-b28f-f6205609afdd)

## Object Detection Model

The model identifies, locates and segments instances of specific objects within images or videos, such as people, vehicles, or animals. It is an RTMDet model trained for obstruction detection, focusing on detecting eight specific classes, including potholes.

  
 ![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/79241d07-2d6f-4184-bc33-85f682d33406)


## Lane Detection Model

  
Lane line detection has been attempted  using computer vision by leveraging techniques such as Adaptive Thresholding and Canny Edge detection can help extract these lines from images in real time. By utilizing these advanced methodologies, the system can reduce costs compared to traditional sensor-based approaches like Lidar and Radar.


![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/1a7f6f1c-1dc5-41e9-92dd-a384dd2b85eb)
### Lane Validation Steps
![lane validation](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/1dd80a3e-6e48-441d-84e0-0a0289b6bc7f)




## Installation Guide

### Step 0. Download and install Miniconda <br><br>

### Step 1. Create a conda environment and activate it.
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### Step 3. Install PyTorch and Cuda
On GPU platforms:
```
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit 
```
On CPU platforms:
```
conda install pytorch torchvision cpuonly -c pytorch
```

### Step 4. Install MMEngine and MMCV using MIM.
```
pip install -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.0" 
```

### Step 5. Install MMDetection.
```
mim install mmdet
```

### Step 6. Install Flask and run the app
```
pip install flask 
flask --app app run
```

## Guide: https://mmdetection.readthedocs.io/en/3.x/get_started.html
