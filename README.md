# Pothole and Lane Detection App

## Introduction

The app aims to showcase a system that accurately detects obstacles and lane  lines using video input from dashboard of a  car by leveraging the RTMdet model of [mmdetection library](https://github.com/open-mmlab/mmdetection) for object detection and lane detection using computer vison techniques. 

The idea was inspired by the fact that most existing road datasets are mainly recorded in Western countries and do not account for challenges like waterlogged potholes and road defects commonly faced in India hence we recorded our own dataset targeting roads in the state of goa, India.The model is trained on a manually collected Goan road dataset spanning over 130KMs and additional online datasets. 

Another feature is that instead of relying on expensive sensors like Lidar and Radar, we have demostrated how object detection in autonomous vehicles can be implemented more affordable by leveraging Deep Learning models and Computer Vision techniques.


![landing_section](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/30ca1875-7684-4bb9-a1ef-c1972b8dd93e)
![WhatsApp Image 2023-05-05 at 19 10 43](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/faa1b83a-e468-4322-a350-bea691d7fe98)
![demo_uploaded_images](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/472864da-ea58-4053-8598-14ca781d8d4d)



https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/8c00a1c7-6ba2-4ce3-b52d-bfd636c37a86




![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/100c9c2e-4209-43cc-ad9d-813c9fa3bc13)




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

### Guide: https://mmdetection.readthedocs.io/en/3.x/get_started.html



## Basic Flow of Model

  
 ![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/6d19f14c-9669-4e8d-9dc2-32a1880332e2)


## Object Detection Model

The model identifies, locates and segments instances of specific objects within images or videos, such as people, vehicles, or animals. It is an RTMDet model trained for obstruction detection, focusing on detecting eight specific classes, including potholes.

  
![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/b46acaa0-9518-4308-8f87-0ca6efbf59c1)



## Lane Detection Model

  
Lane line detection has been attempted  using computer vision by leveraging techniques such as Adaptive Thresholding and Canny Edge detection can help extract these lines from images in real time. By utilizing these advanced methodologies, the system can reduce costs compared to traditional sensor-based approaches like Lidar and Radar.


![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/c47e4071-1aa0-48c1-83e9-e46580f463af)

### Lane Validation Steps
![image](https://github.com/prabhakar-dessai/pothole-detection-app/assets/61088215/4ce45e1d-3b4b-4bd1-a878-404dd81c6db9)





