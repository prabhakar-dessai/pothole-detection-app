# Pothole Detection App

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
