# pothole-detection-app

create an enviro n activate it
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

pip install flask
flask --app app run

guide: https://mmdetection.readthedocs.io/en/3.x/get_started.html
