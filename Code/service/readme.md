# 输入头发分割图，输出头发方向图，人体模型深度图，头发深度图
运行docker
docker run --gpus all --name strand2d-server -d  -p 50084:50084 -w /app -v "$(pwd):/app" hair-base    bash /app/Code/service/run_services.sh

export PYTHONPATH=~/NeuralHDHair/Code:~/NeuralHDHair:$PYTHONPATH
python Code/service/service.py