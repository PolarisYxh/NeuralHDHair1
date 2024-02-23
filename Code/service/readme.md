# 模块介绍
输入原图，经过StepNet模型，输出粗糙的方向图+分割图，用于给segmentanything找点后，经过segmentanything接口，输出头发分割图；
经过OriginStepNet模型和深度图模型，输入头发分割图，最终输出头发方向图，人体模型深度图，头发深度图，变换后的头发原图，相机内外参等信息

## 作为接口使用
运行docker
docker run --gpus all --name strand2d-server -d  -p 50084:50084 -w /app -v "$(pwd):/app" hair-base    bash /app/Code/service/run_services.sh

export PYTHONPATH=~/NeuralHDHair/Code:~/NeuralHDHair:$PYTHONPATH
python Code/service/service.py