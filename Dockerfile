# docker build -t face-parsing-base . #python3.7
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" | tee /etc/apt/sources.list.d/cuda.list
#更换为阿里源
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-updates multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security universe\n\
    deb http://mirrors.aliyun.com/ubuntu/ focal-security multiverse\n"\ > /etc/apt/sources.list
# 安装Python和pip
# RUN apt-get update && \
#     apt-get install -y python3.8 python3-pip git
# RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /app
# COPY . .

RUN mkdir ~/.pip&&echo '[global]\ntimeout = 6000\nindex-url = http://pypi.douban.com/simple\ntrusted-host = pypi.douban.com EOF' > ~/.pip/pip.conf
RUN pip3 install opencv-python-headless matplotlib onnxruntime numpy Flask requests flask_cors 
RUN pip3 install tensorboard scipy
#python3 -m pip install  torch torchvision 安装卡死，改用指定网站下载安装
# RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# for opencv-python
#RUN apt install -y libgl1-mesa-glx libglib2.0-dev vim
#RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt clean & rm -rf /app & rm -rf ~/.cache/pip/*

